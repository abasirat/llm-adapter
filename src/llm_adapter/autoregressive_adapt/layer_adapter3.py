import torch
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
)
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

    def forward(self, x):
        W = self.A @ self.B
        return torch.nn.functional.linear(x, W, self.bias)


class LowDimQKMultiHeadAttention(torch.nn.Module):
    """
    Q: input_dim_q
    K: input_dim_kv
    V: input_dim_v

    Q and K are projected to low-dimensional qk_dim.
    V is NOT projected.
    After per-head attention, head outputs are concatenated and projected:
        (num_heads * input_dim_v) -> output_dim
    """

    def __init__(
        self,
        input_dim_q,
        input_dim_kv,
        input_dim_v,
        output_dim,
        qk_dim,
        num_heads,
        dropout=0.1,
        bias=True,
        temperature=1.0,
        out_rank=None,
    ):
        super().__init__()

        assert qk_dim % num_heads == 0, "qk_dim must be divisible by num_heads"

        self.input_dim_q = input_dim_q
        self.input_dim_kv = input_dim_kv
        self.input_dim_v = input_dim_v
        self.output_dim = output_dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads
        self.head_qk_dim = qk_dim // num_heads
        self.temperature = temperature

        self.q_proj = torch.nn.Linear(input_dim_q, qk_dim, bias=bias)
        self.k_proj = torch.nn.Linear(input_dim_kv, qk_dim, bias=bias)

        concat_value_dim = num_heads * input_dim_v
        if out_rank is not None:
            self.out_proj = LowRankLinear(concat_value_dim, output_dim, out_rank, bias=bias)
        else:
            self.out_proj = torch.nn.Linear(concat_value_dim, output_dim, bias=bias)

        self.attn_dropout = torch.nn.Dropout(dropout)

    def forward(self, Q, K, V, key_padding_mask=None, need_weights=False):
        """
        Q: (B, Tq, Dq)
        K: (B, Tk, Dk)
        V: (B, Tk, Dv)

        Returns:
            output: (B, Tq, output_dim)
            attn_weights: (B, H, Tq, Tk) or None
        """
        B, Tq, _ = Q.shape
        Tk = K.size(1)
        Dv = V.size(-1)

        Q_low = self.q_proj(Q).view(B, Tq, self.num_heads, self.head_qk_dim).transpose(1, 2)
        K_low = self.k_proj(K).view(B, Tk, self.num_heads, self.head_qk_dim).transpose(1, 2)

        # No V projection: repeat full value vector across heads
        V_full = V.unsqueeze(1).expand(B, self.num_heads, Tk, Dv)

        scale = self.head_qk_dim ** 0.5
        scores = torch.matmul(Q_low, K_low.transpose(-2, -1)) / scale / self.temperature
        scores = scores.masked_fill(torch.isnan(scores), float("-inf"))

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, V_full)  # (B, H, Tq, Dv)
        output = output.transpose(1, 2).contiguous().view(B, Tq, self.num_heads * Dv)
        output = self.out_proj(output)  # (B, Tq, output_dim)

        if not need_weights:
            attn_weights = None

        return output, attn_weights


class LayerAdapter(torch.nn.Module):
    def __init__(
        self,
        encoder,
        need_weights=False,
        dropout=0.1,
        num_aggregation_layers=None,
        prefix_length=10,
        qk_dim=32,
        num_attention_heads=4,
        attention_temperature=2.0,
        out_rank=None,
    ):
        super().__init__()

        hs = encoder.config.hidden_size
        nh = encoder.config.n_head
        nl = encoder.config.n_layer if num_aggregation_layers is None else num_aggregation_layers
        head_dim = hs // nh
        mlp_dim = 4 * hs  # GPT-2 style

        self.prefix_length = prefix_length
        self.head_dim = head_dim
        self.full_n_layer = encoder.config.n_layer
        self.agg_n_layer = nl

        self.encoder = encoder
        self.config = encoder.config
        self.encoder.eval()

        self.need_weights = need_weights
        self.num_aggregation_layers = num_aggregation_layers

        self.token_layer_attention = LowDimQKMultiHeadAttention(
            input_dim_q=hs,       # query from residual stream
            input_dim_kv=mlp_dim, # keys from c_fc activations
            input_dim_v=mlp_dim,  # values from c_fc activations
            output_dim=hs,        # final output back to hidden size
            qk_dim=qk_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            temperature=attention_temperature,
            out_rank=out_rank,
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.mlp_c_fc_activations = []

        if self.encoder.name_or_path.startswith("gpt"):
            self.hooks = self.hook_mlp_c_fc_in_gpt()
            self.forward = self.forward_gpt
        elif self.encoder.name_or_path.startswith("bert"):
            raise NotImplementedError("BERT-style models not implemented yet. GPT-style models only for now.")
        else:
            raise NotImplementedError("this implementation works only with standard gpt and bert families from huggingface")

        self.input_dropout = torch.nn.Dropout(dropout)
        self.output_dropout = torch.nn.Dropout(dropout)

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

        self.query_layer_norm = torch.nn.LayerNorm(hs)
        self.kv_layer_norm = torch.nn.LayerNorm(mlp_dim)

    def hook_mlp_c_fc_in_gpt(self):
        def make_hook():
            def hook(module, inputs, output):
                if getattr(module, "is_tailor_block", False):
                    return
                self.mlp_c_fc_activations.append(output)
            return hook

        hooks = []
        for block in self.encoder.h:
            if getattr(block, "is_tailor_block", False):
                continue
            h = block.mlp.c_fc.register_forward_hook(make_hook())
            hooks.append(h)
        return hooks

    def build_prefix_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        if self.prefix_length <= 0:
            return None

        cache = DynamicCache()
        for layer_idx in range(self.full_n_layer):
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

        return past_key_values

    def update_layer_attention_metrics(self, attn_weights):
        if attn_weights is None:
            return

        attn_weights = attn_weights.detach().cpu()
        self.layer_attention_metrics["sum_attention_to_each_layer"] += attn_weights.sum(dim=(0, 1))
        self.layer_attention_metrics["sum_attention_squared_to_each_layer"] += (attn_weights ** 2).sum(dim=(0, 1))
        self.layer_attention_metrics["num_aggregation_layers"] = attn_weights.size(-1)
        self.layer_attention_metrics["num_tokens_aggregated"] += attn_weights.size(0) * attn_weights.size(1)

        avg_attention = self.layer_attention_metrics["sum_attention_to_each_layer"] / self.layer_attention_metrics["num_tokens_aggregated"]
        avg_attention_squared = self.layer_attention_metrics["sum_attention_squared_to_each_layer"] / self.layer_attention_metrics["num_tokens_aggregated"]
        std_attention = torch.sqrt(avg_attention_squared - avg_attention ** 2 + 1e-8)
        entropy_of_attention = -torch.sum(avg_attention * torch.log(avg_attention + 1e-8))

        self.layer_attention_metrics["avg_attention_to_each_layer"] = avg_attention
        self.layer_attention_metrics["std_attention_to_each_layer"] = std_attention
        self.layer_attention_metrics["entropy_of_layer_attention"] = entropy_of_attention

    def aggregator(self, base_hidden_state, c_fc_activations, key_padding_mask=None):
        """
        base_hidden_state: (B, T, D)
        c_fc_activations:  (B, T, 4D, L)
        """
        inputs = self.input_dropout(c_fc_activations)

        # Query from final hidden state
        Q = self.query_layer_norm(base_hidden_state)  # (B, T, D)

        # Per-token stack over layers: (B, T, 4D, L) -> (B*T, L, 4D)
        KV = inputs.permute(0, 1, 3, 2).contiguous().view(-1, inputs.size(-1), inputs.size(2))
        K = self.kv_layer_norm(KV)
        V = self.kv_layer_norm(KV)

        Q_bt = Q.contiguous().view(-1, 1, Q.size(-1))

        attn_output, attn_weights = self.token_layer_attention(
            Q_bt,
            K,
            V,
            key_padding_mask=None,
            need_weights=self.need_weights,
        )

        delta = attn_output.squeeze(1).view(base_hidden_state.size(0), base_hidden_state.size(1), -1)
        r = base_hidden_state + self.adapter_scale * self.output_dropout(delta)

        if attn_weights is not None:
            attn_weights = attn_weights.mean(dim=1)   # (B*T, 1, L)
            attn_weights = attn_weights.squeeze(1)    # (B*T, L)
            attn_weights = attn_weights.view(base_hidden_state.size(0), base_hidden_state.size(1), inputs.size(-1))

        if key_padding_mask is not None:
            r = r.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            if attn_weights is not None:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return r, attn_weights

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

        self.mlp_c_fc_activations = []

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
                return_dict=return_dict if return_dict is not None else True,
                cache_position=cache_position,
            )

        assert len(self.mlp_c_fc_activations) == self.config.n_layer, \
            f"Expected {self.config.n_layer} c_fc activations, got {len(self.mlp_c_fc_activations)}"

        # (B, T, 4D, L)
        c_fc = torch.stack(self.mlp_c_fc_activations, dim=-1)

        if self.num_aggregation_layers is not None:
            c_fc = c_fc[..., -self.num_aggregation_layers:]

        if attention_mask is not None:
            key_padding_mask = ~attention_mask[:, -c_fc.size(1):].bool()
        else:
            key_padding_mask = None

        aggregated_hidden_state, layer_token_attention = self.aggregator(
            base_hidden_state=encoder_outputs.last_hidden_state,
            c_fc_activations=c_fc,
            key_padding_mask=key_padding_mask,
        )

        self.mlp_c_fc_activations = []

        if layer_token_attention is not None:
            self.update_layer_attention_metrics(layer_token_attention)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=aggregated_hidden_state,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )