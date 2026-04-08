import pdb

import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions

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
    def __init__(self, input_dim, qk_dim, num_heads, v_dim=None, dropout=0.1, bias=True):
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

        scale = self.head_qk_dim ** -0.5
        scores = torch.matmul(Q_low, K_low.transpose(-2, -1)) * scale
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
        ):
        super(LayerAdapter, self).__init__()

        hs = encoder.config.hidden_size
        nh = encoder.config.n_head
        nl = encoder.config.n_layer if num_aggregation_layers is None else num_aggregation_layers

        self.n_layer = nl
        self.prefix_length = prefix_length

        self.encoder = encoder
        self.config = encoder.config

        self.encoder.eval() # Set encoder to evaluation mode to disable dropout and other training-specific layers. It will speed up inference and reduce memory usage. We will enable grad only for the adapters and lm_head.

        self.need_weights = need_weights
        self.num_aggregation_layers = num_aggregation_layers

        #self.val_projection = torch.nn.Linear(hs, 32)
        #self.key_projection = torch.nn.Linear(hs, 32)

        #self.token_layer_attention = torch.nn.MultiheadAttention(
        #        embed_dim = hs, 
        #        num_heads = 4, 
        #        kdim = 32,
        #        vdim = 32,
        #        batch_first = True,
        #        dropout = dropout,
        #)
        self.token_layer_attention = LowDimQKMultiHeadAttention(
                input_dim = hs,
                qk_dim = 128,
                num_heads = 4,
                dropout = dropout,
        )

        for param in self.encoder.parameters(): 
            param.requires_grad = False
        
        self.before_mlp_activations = []
        if self.encoder.name_or_path.startswith("gpt"):
            self.hooks = self.hook_before_each_mlp_in_gpt()
            self.forward = self.forward_gpt
        elif self.encoder.name_or_path.startswith("bert"):
            self.hooks = self.hook_before_each_mlp_in_bert()
            self.forward = self.forward_bert
        else:
            print("this implementation works only with standard gpt and bert families from huggingface")
            raise NotImplementedError
        
        self.output_dropout = torch.nn.Dropout(dropout)

        self.pre_mlp_linear_transforms = torch.nn.ModuleDict({
            f"layer_{i}": LowRankLinear(hs, hs, rank=8) for i in range(nl)
            #f"layer_{i}": torch.nn.Linear(hs, hs) for i in range(nl)
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

        # Prefix tokens for prefix-tuning style adaptation. Shape (prefix_length, hidden_size)
        if prefix_length > 0:
            self.prefix_embeddings = torch.nn.Parameter(
                torch.randn(prefix_length, hs) * 0.02
            )  # (P, D)
        else:
            self.prefix_embeddings = None

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
    
    def get_prefix_states(self, batch_size):
        """Generate prefix key-value states for all layers"""
        prefix_states = []
        
        for layer_idx in range(self.n_layer):
            # Get prefix embeddings for current layer
            prefix_k = self.prefix_embeddings[layer_idx, 0]  # Key prefix
            prefix_v = self.prefix_embeddings[layer_idx, 1]  # Value prefix
            
            # Expand for batch size
            prefix_k = prefix_k.unsqueeze(0).expand(batch_size, -1, -1, -1)
            prefix_v = prefix_v.unsqueeze(0).expand(batch_size, -1, -1, -1)
            
            prefix_states.append((prefix_k, prefix_v))
            
        return prefix_states

    def prepend_prefix_embeddings(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        if self.prefix_embeddings is None:
            return input_ids, inputs_embeds, attention_mask, 0

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            inputs_embeds = self.encoder.get_input_embeddings()(input_ids)

        B = inputs_embeds.size(0)
        P = self.prefix_length
        D = inputs_embeds.size(-1)

        prefix = self.prefix_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, P, D)
        inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)       # (B, P+T, D)

        if attention_mask is None:
            attention_mask = torch.ones(
                B, inputs_embeds.size(1), device=inputs_embeds.device, dtype=torch.long
            )
        else:
            prefix_mask = torch.ones(B, P, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return None, inputs_embeds, attention_mask, P
    
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

        
        inputs = hidden_states + torch.stack([self.pre_mlp_linear_transforms[f"layer_{i}"](pre_mlp_activations[..., i]) for i in range(pre_mlp_activations.size(-1))], dim=-1) # shape (B, T, D, L)

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

        with torch.no_grad():
            encoder_outputs =  self.encoder(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict,
                    cache_position=cache_position,
            )

        key_padding_mask = torch.logical_not(attention_mask)
        assert len(self.before_mlp_activations) == self.config.n_layer, \
            f"Expected {self.config.n_layer} pre-MLP activations, got {len(self.before_mlp_activations)}"
        hs = torch.stack(encoder_outputs.hidden_states, -1)[..., 1:] # shape (B, T, D, L)
        ac = torch.stack(self.before_mlp_activations, -1) #  shape (B, T, D, L)
        if self.num_aggregation_layers is not None:
            hs = hs[..., -self.num_aggregation_layers:]
            ac = ac[..., -self.num_aggregation_layers:]
        aggregated_hidden_state, layer_token_attention = self.aggregator(hs, ac, key_padding_mask)
        self.before_mlp_activations = [] # clear stored activations after use

        if output_attentions is not None:
            self.update_layer_attention_metrics(layer_token_attention)

        outputs = BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state = aggregated_hidden_state,
                pooler_output = encoder_outputs.pooler_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
                attentions=encoder_outputs.attentions,
                cross_attentions=encoder_outputs.cross_attentions,
        )

        return outputs
    
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

        original_attention_mask = attention_mask
        
        input_ids, inputs_embeds, attention_mask, prefix_len = self.prepend_prefix_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=input_ids,                  # now None if prefix was added
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

        assert len(self.before_mlp_activations) == self.config.n_layer, \
            f"Expected {self.config.n_layer} pre-MLP activations, got {len(self.before_mlp_activations)}"

        # hidden states now include prefix positions, so remove them before aggregation
        hs = torch.stack(encoder_outputs.hidden_states, -1)[..., 1:]   # (B, T+P, D, L)
        ac = torch.stack(self.before_mlp_activations, -1)              # (B, T+P, D, L)

        if prefix_len > 0:
            hs = hs[:, prefix_len:, :, :]
            ac = ac[:, prefix_len:, :, :]

        if self.num_aggregation_layers is not None:
            hs = hs[..., -self.num_aggregation_layers:]
            ac = ac[..., -self.num_aggregation_layers:]

        if original_attention_mask is not None:
            key_padding_mask = ~original_attention_mask.bool()
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