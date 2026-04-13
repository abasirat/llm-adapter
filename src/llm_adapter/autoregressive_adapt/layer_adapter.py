import pdb

import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers.cache_utils import DynamicCache
class LowRankLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True, init_std=0.02):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(out_features, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, in_features))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        torch.nn.init.normal_(self.A, mean=0.0, std=init_std)
        # B already zero

    def forward(self, x):
        W = self.A @ self.B
        return torch.nn.functional.linear(x, W, self.bias)

class LowDimQKMultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, qk_dim, num_heads, v_dim=None, dropout=0.1, bias=True, temperature=1.0):
        super().__init__()

        assert qk_dim % num_heads == 0, "qk_dim must be divisible by num_heads"

        self.input_dim = input_dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads
        self.head_qk_dim = qk_dim // num_heads
        self.v_dim = v_dim if v_dim is not None else input_dim

        assert self.v_dim % num_heads == 0, "v_dim must be divisible by num_heads"
        self.head_v_dim = self.v_dim // num_heads

        self.q_proj = torch.nn.Linear(input_dim, qk_dim, bias=bias)
        self.k_proj = torch.nn.Linear(input_dim, qk_dim, bias=bias)
        self.v_proj = torch.nn.Identity() if self.v_dim == input_dim else torch.nn.Linear(input_dim, self.v_dim, bias=bias)
        self.out_proj = torch.nn.Identity() if self.v_dim == input_dim else torch.nn.Linear(self.v_dim, input_dim, bias=bias)

        self.attn_dropout = torch.nn.Dropout(dropout)

        self.temperature = temperature

    def forward(self, Q, K, V, key_padding_mask=None, need_weights=False):
        """
        Q: (B, Tq, D)
        K: (B, Tk, D)
        V: (B, Tk, Dv_in)

        Returns:
            output: (B, Tq, input_dim)
            attn_weights: (B, num_heads, Tq, Tk) or None
        """
        B, Tq, _ = Q.shape
        Tk = K.size(1)

        Q_low = self.q_proj(Q).view(B, Tq, self.num_heads, self.head_qk_dim).transpose(1, 2)
        K_low = self.k_proj(K).view(B, Tk, self.num_heads, self.head_qk_dim).transpose(1, 2)
        V_full = self.v_proj(V).view(B, Tk, self.num_heads, self.head_v_dim).transpose(1, 2)

        # Q_low, K_low, V_full:
        # (B, H, Tq, dq_head), (B, H, Tk, dq_head), (B, H, Tk, dv_head)

        scale = self.head_qk_dim ** 0.5
        scores = torch.matmul(Q_low, K_low.transpose(-2, -1)) / scale / self.temperature  # (B, H, Tq, Tk)
        scores = scores.masked_fill(torch.isnan(scores), float("-inf"))  # Handle NaNs from zero vectors in Q/K 
        # (B, H, Tq, Tk)

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, V_full)  # (B, H, Tq, dv_head)
        output = output.transpose(1, 2).contiguous().view(B, Tq, self.v_dim)
        output = self.out_proj(output)  # (B, Tq, input_dim)

        if not need_weights:
            attn_weights = None

        return output, attn_weights
class LayerAdapter(torch.nn.Module):
    def __init__(self, 
            encoder, 
            need_weights=False,
            dropout=0.1,
            num_aggregation_layers=None,
            prefix_length=10,
            adjust_pre_mlps=False,
        ):
        super(LayerAdapter, self).__init__()

        hs = encoder.config.hidden_size
        nh = encoder.config.n_head
        nl = encoder.config.n_layer if num_aggregation_layers is None else num_aggregation_layers
        head_dim = hs // nh

        self.prefix_length = prefix_length
        self.head_dim = head_dim
        self.full_n_layer = encoder.config.n_layer
        self.agg_n_layer = nl

        self.encoder = encoder
        self.config = encoder.config

        self.encoder.eval() # Set encoder to evaluation mode to disable dropout and other training-specific layers. It will speed up inference and reduce memory usage. We will enable grad only for the adapters and lm_head.

        self.need_weights = need_weights
        self.num_aggregation_layers = num_aggregation_layers

        self.adjust_pre_mlps = adjust_pre_mlps

        self.token_layer_attention = LowDimQKMultiHeadAttention(
                input_dim = hs,
                qk_dim = 32,
                num_heads = 4,
                dropout = dropout,
                temperature = 2.0,
        )

        for param in self.encoder.parameters(): 
            param.requires_grad = False
        
        self.before_mlp_activations = []
        if self.encoder.name_or_path.startswith("gpt"):
            self.hooks = self.hook_before_each_mlp_in_gpt() if adjust_pre_mlps else []
            self.forward = self.forward_gpt
        elif self.encoder.name_or_path.startswith("bert"):
            self.hooks = self.hook_before_each_mlp_in_bert()
            self.forward = self.forward_bert
        else:
            print("this implementation works only with standard gpt and bert families from huggingface")
            raise NotImplementedError
        
        self.input_dropout = torch.nn.Dropout(dropout)
        self.output_dropout = torch.nn.Dropout(dropout)

        self.pre_mlp_linear_transforms = None
        if adjust_pre_mlps:
            self.pre_mlp_linear_transforms = torch.nn.ModuleDict({
                f"layer_{i}": LowRankLinear(hs, hs, rank=8) for i in range(self.config.n_layer, self.config.n_layer - nl, -1)
                #f"layer_{i}": torch.nn.Linear(hs, hs) for i in range(self.config.n_layer, self.config.n_layer - nl, -1)
            })

        self.adapter_scale = torch.nn.Parameter(torch.tensor(0.0))

        self.layer_attention_metrics = {
            "avg_attention_to_each_layer": torch.zeros(nl),
            "std_attention_to_each_layer": torch.zeros(nl),
            "entropy_of_layer_attention": torch.zeros(nl),
            "num_aggregation_layers": nl,
            "sum_attention_to_each_layer": torch.zeros(nl),
            "sum_attention_squared_to_each_layer": torch.zeros(nl),
            "num_tokens_aggregated": 0,
        }

        # # Trainable KV prefix for all transformer layers (not just aggregated ones).
        if prefix_length > 0:
            self.prefix_key = torch.nn.Parameter(
                torch.randn(self.full_n_layer, nh, prefix_length, head_dim) * 0.02
            )
            self.prefix_value = torch.nn.Parameter(
                torch.randn(self.full_n_layer, nh, prefix_length, head_dim) * 0.02
            )
        else:
            self.prefix_key = None
            self.prefix_value = None
        
        self.layer_norm = torch.nn.LayerNorm(hs)

    def print_trainable_parameters(self):
        """
            Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(f"layer-wise adapter's trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")
    
    def hook_before_each_mlp_in_gpt(self):
        def make_mlp_input_hook():
            def hook(module, inputs):
                if getattr(module, "is_tailor_block", False):
                    return
                self.before_mlp_activations.append(inputs[0])
            return hook

        hooks = []
        for block in self.encoder.h:
            if getattr(block, "is_tailor_block", False):
                continue
            h = block.mlp.register_forward_pre_hook(make_mlp_input_hook())
            hooks.append(h)
        return hooks
    
    def hook_before_each_mlp_in_bert(self):
        raise NotImplementedError("BERT-style models not implemented yet. GPT-style models only for now.")
    
    def build_prefix_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """
        Returns a DynamicCache prefilled with learned prefix KV.
        """
        if self.prefix_length <= 0:
            return None

        cache = DynamicCache()

        for layer_idx in range(self.full_n_layer):
            # (H, P, Dh) -> (B, H, P, Dh)
            k = self.prefix_key[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
            v = self.prefix_value[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)

            k = k.to(device=device, dtype=dtype)
            v = v.to(device=device, dtype=dtype)

            cache.update(k, v, layer_idx)

        return cache

    def maybe_add_prefix_cache(self, past_key_values, batch_size, device, dtype):
        if self.prefix_length <= 0:
            return past_key_values

        if past_key_values is None:
            return self.build_prefix_cache(batch_size, device, dtype)

        if hasattr(past_key_values, "get_seq_length"):
            if past_key_values.get_seq_length() > 0:
                return past_key_values
            return self.build_prefix_cache(batch_size, device, dtype)

        # Legacy tuple/list cache: assume it is already meaningful and do not overwrite it
        return past_key_values
    
    def update_layer_attention_metrics(self, attn_weights):
        # attn_weights shape: (B, T, L)
        if attn_weights is None:
            return
        
        attn_weights = attn_weights.detach().cpu()
        self.layer_attention_metrics["sum_attention_to_each_layer"] += attn_weights.sum(dim=(0, 1))
        self.layer_attention_metrics["sum_attention_squared_to_each_layer"] += (attn_weights ** 2).sum(dim=(0, 1))
        self.layer_attention_metrics["num_aggregation_layers"] = attn_weights.size(-1)
        self.layer_attention_metrics["num_tokens_aggregated"] += attn_weights.size(0) * attn_weights.size(1)

        # Compute average and std attention to each layer
        avg_attention = self.layer_attention_metrics["sum_attention_to_each_layer"] / self.layer_attention_metrics["num_tokens_aggregated"]
        avg_attention_squared = self.layer_attention_metrics["sum_attention_squared_to_each_layer"] / self.layer_attention_metrics["num_tokens_aggregated"]
        std_attention = torch.sqrt(avg_attention_squared - avg_attention ** 2 + 1e-8)
        entropy_of_attention = -torch.sum(avg_attention * torch.log(avg_attention + 1e-8))
        self.layer_attention_metrics["avg_attention_to_each_layer"] = avg_attention
        self.layer_attention_metrics["std_attention_to_each_layer"] = std_attention
        self.layer_attention_metrics["entropy_of_layer_attention"] = entropy_of_attention
        
    def aggregator(self, hidden_states, pre_mlp_activations, key_padding_mask=None):
        """
        hidden_states: (B, T, D, L)
            B = batch
            T = sequence length
            D = hidden size
            L = number of hidden-state layers being aggregated
        
        pre_mlp_activations: (B, T, D, L)

        Returns:
            r:   (B, T, D)      aggregated representation
            w:   (B, T, L)      per-token layer weights
        """
        if self.adjust_pre_mlps:
            nl = pre_mlp_activations.size(-1)
            pre_mlps = torch.stack([self.pre_mlp_linear_transforms[f"layer_{i}"](pre_mlp_activations[..., i - self.config.n_layer + nl - 1]) for i in range(self.config.n_layer, self.config.n_layer - nl, -1)], dim=-1) # shape (B, T, D, L)
        else:
            pre_mlps = 0
        inputs = hidden_states + pre_mlps # shape (B, T, D, L)
        inputs = self.input_dropout(inputs)

        # Query from the same token position, averaged over layers
        # Q: (B, T, D)
        Q = inputs[..., -1] # use the top layer as query
        # Keys/values are the layer representations for the SAME token only.
        # Rearrange to: (B*T, ??
        KV = inputs.permute(0, 1, 3, 2).contiguous().view(-1, inputs.size(-1), inputs.size(2))

        K = KV
        V = KV

        # Query per token: (B*T, 1, D)
        Q_bt = Q.contiguous().view(-1, 1, Q.size(-1))

        Q_bt = self.layer_norm(Q_bt)
        K = self.layer_norm(K)
        V = self.layer_norm(V)

        if key_padding_mask is not None:
            # Attending over layers, not tokens, so token padding mask is for the query side.
            # It should NOT be used as a key mask here because Tk = L, not sequence length.
            layer_key_padding_mask = None
        else:
            layer_key_padding_mask = None

        # Attend from each token to its own layer stack only.
        # No token-token mixing => causal by construction.
        attn_output, attn_weights = self.token_layer_attention(
            Q_bt, K, V, key_padding_mask=layer_key_padding_mask, need_weights=self.need_weights
        )

        #r = attn_output.squeeze(1).view(inputs.size(0), inputs.size(1), -1)
        base = hidden_states[..., -1]
        delta = attn_output.squeeze(1).view(inputs.size(0), inputs.size(1), -1)
        r = base + self.adapter_scale * self.output_dropout(delta)

        if attn_weights is not None:
            # attn_weights: (B*T, H, 1, L)
            attn_weights = attn_weights.mean(dim=1)  # (B*T, 1, L)
            attn_weights = attn_weights.squeeze(1)   # (B*T, L)
            attn_weights = attn_weights.view(inputs.size(0), inputs.size(1), inputs.size(-1))  # (B, T, L)

        if key_padding_mask is not None:
            r = r.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            if attn_weights is not None:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return r, attn_weights

    def forward_bert(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        )-> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        raise NotImplementedError("BERT-style models not implemented yet. GPT-style models only for now.")
    
    def forward_gpt(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            batch_size = input_ids.size(0)
            device = input_ids.device
            dtype = self.encoder.wte.weight.dtype
        else:
            batch_size = inputs_embeds.size(0)
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype

        past_key_values = self.maybe_add_prefix_cache(
            past_key_values=past_key_values,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        is_first_prefill_with_prefix = (
            attention_mask is not None
            and self.prefix_length > 0
            and past_key_values is not None
            and hasattr(past_key_values, "get_seq_length")
            and past_key_values.get_seq_length() == self.prefix_length
        )

        if is_first_prefill_with_prefix:
            prefix_attention_mask = torch.ones(
                batch_size,
                self.prefix_length,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        assert len(self.before_mlp_activations) == self.config.n_layer or not self.adjust_pre_mlps, \
            f"Expected {self.config.n_layer} pre-MLP activations, got {len(self.before_mlp_activations)}"

        hs = torch.stack(encoder_outputs.hidden_states, -1)[..., 1:]
        ac = torch.stack(self.before_mlp_activations, -1) if self.adjust_pre_mlps else None

        if self.num_aggregation_layers is not None:
            hs = hs[..., -self.num_aggregation_layers:]
            ac = ac[..., -self.num_aggregation_layers:] if ac is not None else None

        if attention_mask is not None:
            # key_padding_mask = ~attention_mask.bool()
            # If extended attention_mask with prefix ones above,
            # then use only the real-token part here:
            key_padding_mask = ~attention_mask[:, -hs.size(1):].bool()
        else:
            key_padding_mask = None

        aggregated_hidden_state, layer_token_attention = self.aggregator(hs, ac, key_padding_mask)
        self.before_mlp_activations = []

        if layer_token_attention is not None:
            self.update_layer_attention_metrics(layer_token_attention)

        outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=aggregated_hidden_state,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

        return outputs