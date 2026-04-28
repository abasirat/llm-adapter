import pdb

import torch
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import (
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

        #self.v_proj = torch.nn.Sequential(
        #    torch.nn.Linear(input_dim_v, 4 * input_dim_v, bias=bias),
        #    torch.nn.GELU(),
        #    torch.nn.Linear(4 * input_dim_v, input_dim_v, bias=bias),
        #)

        concat_value_dim = num_heads * input_dim_v
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

        #V_proj = self.v_proj(V)
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

class VariationalDeltaHead(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        use_mean_in_eval=True,
        kl_reduction="mean",
        min_std=1e-6,
    ):
        super().__init__()

        if kl_reduction not in {"mean", "sum"}:
            raise ValueError(f"kl_reduction must be 'mean' or 'sum', got {kl_reduction!r}")

        self.use_mean_in_eval = use_mean_in_eval
        self.kl_reduction = kl_reduction
        self.min_std = min_std

        self.mean_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.logvar_proj = torch.nn.Linear(hidden_size, hidden_size)

        self.last_kl_loss = None
        self.last_delta_loss = None
        self.last_mu = None
        self.last_std = None

    def forward(self, deterministic_delta, key_padding_mask=None):
        """
        deterministic_delta: (B, T, D)
        key_padding_mask:    (B, T), True for padding tokens
        """
        mu = self.mean_proj(deterministic_delta)
        logvar = self.logvar_proj(deterministic_delta)

        if key_padding_mask is not None:
            mu = mu.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            logvar = logvar.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        std = torch.exp(0.5 * logvar).clamp_min(self.min_std)

        if self.training:
            eps = torch.randn_like(std)
            delta = mu + std * eps
        else:
            delta = mu if self.use_mean_in_eval else mu + std * torch.randn_like(std)

        kl_per_dim = 0.5 * (mu.pow(2) + torch.exp(logvar) - 1.0 - logvar)

        if key_padding_mask is not None:
            valid = (~key_padding_mask).unsqueeze(-1).to(kl_per_dim.dtype)
            kl_per_dim = kl_per_dim * valid
            denom = (valid.sum() * kl_per_dim.size(-1)).clamp_min(1.0)
        else:
            denom = torch.tensor(
                kl_per_dim.numel(),
                device=kl_per_dim.device,
                dtype=kl_per_dim.dtype,
            )

        if self.kl_reduction == "sum":
            kl_loss = kl_per_dim.sum()
            delta_loss = delta.pow(2).sum()
        else:
            kl_loss = kl_per_dim.sum() / denom
            delta_loss = delta.pow(2).mean()

        self.last_mu = mu
        self.last_std = std
        self.last_kl_loss = kl_loss
        self.last_delta_loss = delta_loss

        return delta

    def get_kl_loss(self):
        if self.last_kl_loss is None:
            raise RuntimeError("No forward pass has been run yet.")
        return self.last_kl_loss

    def get_delta_loss(self):
        if self.last_delta_loss is None:
            raise RuntimeError("No forward pass has been run yet.")
        return self.last_delta_loss

    def get_variational_stats(self):
        if self.last_mu is None or self.last_std is None:
            raise RuntimeError("No forward pass has been run yet.")
        return self.last_mu, self.last_std


class LayerAttentionAggregator(torch.nn.Module):
    VALID_QUERY_SOURCES = {"final_hidden", "top_repr"}

    def __init__(
        self,
        hidden_size,
        rep_dim,
        num_layers,
        need_weights=False,
        dropout=0.1,
        qk_dim=32,
        v_dim=None,
        num_attention_heads=4,
        attention_temperature=2.0,
        query_source="final_hidden",
    ):
        super().__init__()

        if query_source not in self.VALID_QUERY_SOURCES:
            raise ValueError(
                f"query_source must be one of {sorted(self.VALID_QUERY_SOURCES)}, "
                f"got {query_source!r}"
            )

        self.hidden_size = hidden_size
        self.rep_dim = rep_dim
        self.num_layers = num_layers
        self.need_weights = need_weights
        self.query_source = query_source

        self.v_dim = v_dim if v_dim is not None else rep_dim

        self.linear_aligners = torch.nn.ModuleList([
            torch.nn.Linear(rep_dim, self.v_dim) for _ in range(num_layers)
        ])

        self.input_dropout = torch.nn.Dropout(dropout)
        self.output_dropout = torch.nn.Dropout(dropout)

        self.query_layer_norm = torch.nn.LayerNorm(hidden_size)
        self.kv_layer_norm = torch.nn.LayerNorm(self.v_dim)

        self.query_from_repr_proj = (
            torch.nn.Linear(self.v_dim, hidden_size)
            if query_source == "top_repr" and self.v_dim != hidden_size
            else torch.nn.Identity()
        )

        self.token_layer_attention = LowDimQKMultiHeadAttention(
            input_dim_q=hidden_size,
            input_dim_kv=self.v_dim,
            input_dim_v=self.v_dim,
            output_dim=hidden_size,
            qk_dim=qk_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            temperature=attention_temperature,
        )

    def align_representations(self, layer_representations):
        """
        layer_representations: tuple/list of L tensors, each (B, T, R)

        returns:
            reps: (B, T, V, L)
        """
        if len(layer_representations) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer representations, "
                f"got {len(layer_representations)}"
            )

        aligned = [
            self.linear_aligners[i](rep)
            for i, rep in enumerate(layer_representations)
        ]

        return torch.stack(aligned, dim=-1)

    def forward(
        self,
        base_hidden_state,
        layer_representations,
        key_padding_mask=None,
    ):
        """
        base_hidden_state:     (B, T, D)
        layer_representations: tuple/list of L tensors, each (B, T, R)

        returns:
            deterministic_delta: (B, T, D)
            attn_weights:        (B, T, L) or None
        """
        reps = self.align_representations(layer_representations)
        inputs = self.input_dropout(reps)

        if self.query_source == "final_hidden":
            Q = self.query_layer_norm(base_hidden_state)

        elif self.query_source == "top_repr":
            top_repr = inputs[..., -1]
            top_repr = self.query_from_repr_proj(top_repr)
            Q = self.query_layer_norm(top_repr)

        else:
            raise ValueError(f"Unknown query_source: {self.query_source}")

        # (B, T, V, L) -> (B*T, L, V)
        KV = (
            inputs
            .permute(0, 1, 3, 2)
            .contiguous()
            .view(-1, inputs.size(-1), inputs.size(2))
        )

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

        deterministic_delta = attn_output.squeeze(1).view(
            base_hidden_state.size(0),
            base_hidden_state.size(1),
            -1,
        )

        deterministic_delta = self.output_dropout(deterministic_delta)

        if attn_weights is not None:
            attn_weights = attn_weights.mean(dim=1)
            attn_weights = attn_weights.squeeze(1)
            attn_weights = attn_weights.view(
                base_hidden_state.size(0),
                base_hidden_state.size(1),
                inputs.size(-1),
            )

            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(-1),
                    0.0,
                )

        return deterministic_delta, attn_weights


class MeanLayerAggregator(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        rep_dim,
        num_layers,
        dropout=0.1,
        v_dim=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.rep_dim = rep_dim
        self.num_layers = num_layers
        self.v_dim = v_dim if v_dim is not None else rep_dim

        self.linear_aligners = torch.nn.ModuleList([
            torch.nn.Linear(rep_dim, self.v_dim) for _ in range(num_layers)
        ])

        self.dropout = torch.nn.Dropout(dropout)
        self.output_proj = torch.nn.Linear(self.v_dim, hidden_size)

    def forward(
        self,
        base_hidden_state,
        layer_representations,
        key_padding_mask=None,
    ):
        """
        returns:
            deterministic_delta: (B, T, D)
            attn_weights:        None
        """
        if len(layer_representations) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer representations, "
                f"got {len(layer_representations)}"
            )

        aligned = [
            self.linear_aligners[i](rep)
            for i, rep in enumerate(layer_representations)
        ]

        reps = torch.stack(aligned, dim=-1)  # (B, T, V, L)
        reps = self.dropout(reps)

        pooled = reps.mean(dim=-1)  # (B, T, V)
        deterministic_delta = self.output_proj(pooled)

        return deterministic_delta, None

class ConcatLayerAggregator(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        rep_dim,
        num_layers,
        dropout=0.1,
        v_dim=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.rep_dim = rep_dim
        self.num_layers = num_layers
        self.v_dim = v_dim if v_dim is not None else rep_dim

        self.linear_aligners = torch.nn.ModuleList([
            torch.nn.Linear(rep_dim, self.v_dim) for _ in range(num_layers)
        ])

        self.dropout = torch.nn.Dropout(dropout)

        self.output_proj = torch.nn.Linear(
            self.num_layers * self.v_dim,
            hidden_size,
        )

    def forward(
        self,
        base_hidden_state,
        layer_representations,
        key_padding_mask=None,
    ):
        """
        base_hidden_state:     (B, T, D)
        layer_representations: tuple/list of L tensors, each (B, T, R)

        returns:
            deterministic_delta: (B, T, D)
            attn_weights:        None
        """
        if len(layer_representations) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer representations, "
                f"got {len(layer_representations)}"
            )

        aligned = [
            self.linear_aligners[i](rep)
            for i, rep in enumerate(layer_representations)
        ]

        # List of L tensors, each (B, T, V)
        # Concatenate across representation dimension:
        # (B, T, V * L)
        concat_reps = torch.cat(aligned, dim=-1)

        concat_reps = self.dropout(concat_reps)

        # (B, T, V * L) -> (B, T, D)
        deterministic_delta = self.output_proj(concat_reps)

        if key_padding_mask is not None:
            deterministic_delta = deterministic_delta.masked_fill(
                key_padding_mask.unsqueeze(-1),
                0.0,
            )

        return deterministic_delta, None

class LayerAdapter(torch.nn.Module):
    """
    representation_type:
        - "pre_mlp"  : input to block.mlp
        - "mid_mlp"  : output of block.mlp.c_fc
        - "post_mlp" : output of block.mlp.c_proj

    query_source:
        - "final_hidden"
        - "top_repr"

    aggregator_type:
        - "attention"
        - "mean"
    """

    VALID_REPRESENTATIONS = {"pre_mlp", "mid_mlp", "post_mlp"}
    VALID_QUERY_SOURCES = {"final_hidden", "top_repr"}
    VALID_AGGREGATORS = {"attention", "mean", "concat"}

    def __init__(
        self,
        encoder,
        need_weights=False,
        dropout=0.1,
        num_aggregation_layers=None,
        prefix_length=10,
        qk_dim=32,
        v_dim=None,
        num_attention_heads=4,
        attention_temperature=2.0,
        representation_type="mid_mlp",
        query_source="final_hidden",
        aggregation_strategy="attention",
        variational_modeling=False,
        variational_use_mean_in_eval=True,
        kl_reduction="mean",
        min_std=1e-6,
    ):
        super().__init__()

        if representation_type not in self.VALID_REPRESENTATIONS:
            raise ValueError(
                f"representation_type must be one of {sorted(self.VALID_REPRESENTATIONS)}, "
                f"got {representation_type!r}"
            )

        if query_source not in self.VALID_QUERY_SOURCES:
            raise ValueError(
                f"query_source must be one of {sorted(self.VALID_QUERY_SOURCES)}, "
                f"got {query_source!r}"
            )

        if aggregation_strategy not in self.VALID_AGGREGATORS:
            raise ValueError(
                f"aggregation_strategy must be one of {sorted(self.VALID_AGGREGATORS)}, "
                f"got {aggregation_strategy!r}"
            )

        hs = encoder.config.hidden_size
        nh = encoder.config.n_head
        nl = encoder.config.n_layer if num_aggregation_layers is None else num_aggregation_layers
        head_dim = hs // nh
        mlp_dim = 4 * hs

        self.encoder = encoder
        self.config = encoder.config
        self.encoder.eval()

        self.hidden_size = hs
        self.mlp_dim = mlp_dim
        self.prefix_length = prefix_length
        self.head_dim = head_dim
        self.full_n_layer = encoder.config.n_layer
        self.num_aggregation_layers = num_aggregation_layers
        self.representation_type = representation_type
        self.query_source = query_source
        self.aggregation_strategy = aggregation_strategy
        self.need_weights = need_weights

        self.rep_dim = self._get_representation_dim(representation_type)

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.layer_representations = []

        if aggregation_strategy == "attention":
            self.aggregator = LayerAttentionAggregator(
                hidden_size=hs,
                rep_dim=self.rep_dim,
                num_layers=nl,
                need_weights=need_weights,
                dropout=dropout,
                qk_dim=qk_dim,
                v_dim=v_dim,
                num_attention_heads=num_attention_heads,
                attention_temperature=attention_temperature,
                query_source=query_source,
            )

        elif aggregation_strategy == "mean":
            self.aggregator = MeanLayerAggregator(
                hidden_size=hs,
                rep_dim=self.rep_dim,
                num_layers=nl,
                dropout=dropout,
                v_dim=v_dim,
            )
        elif aggregation_strategy == "concat":
            self.aggregator = ConcatLayerAggregator(
                hidden_size=hs,
                rep_dim=self.rep_dim,
                num_layers=nl,
                dropout=dropout,
                v_dim=v_dim,
            )

        self.variational_modeling = variational_modeling

        self.variational_head = (
            VariationalDeltaHead(
                hidden_size=hs,
                use_mean_in_eval=variational_use_mean_in_eval,
                kl_reduction=kl_reduction,
                min_std=min_std,
            )
            if variational_modeling
            else None
        )

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

        if self.encoder.name_or_path.startswith("gpt"):
            self.hooks = self._register_gpt_hooks()
            self.forward = self.forward_gpt
        elif self.encoder.name_or_path.startswith("bert"):
            raise NotImplementedError("BERT-style models not implemented yet.")
        else:
            raise NotImplementedError(
                "This implementation works only with standard GPT and BERT families from HuggingFace."
            )
        print(self)

    def _get_representation_dim(self, representation_type):
        if representation_type in {"pre_mlp", "post_mlp"}:
            return self.hidden_size
        if representation_type == "mid_mlp":
            return self.mlp_dim
        raise ValueError(f"Unknown representation_type: {representation_type}")

    def _register_gpt_hooks(self):
        if self.representation_type == "pre_mlp":
            return self._hook_pre_mlp_in_gpt()
        if self.representation_type == "mid_mlp":
            return self._hook_mid_mlp_in_gpt()
        if self.representation_type == "post_mlp":
            return self._hook_post_mlp_in_gpt()
        raise ValueError(f"Unknown representation_type: {self.representation_type}")

    def _hook_pre_mlp_in_gpt(self):
        def make_hook(block):
            def hook(module, inputs):
                if getattr(block, "is_tailor_block", False):
                    return
                self.layer_representations.append(inputs[0])
            return hook

        hooks = []
        for block in self.encoder.h:
            if getattr(block, "is_tailor_block", False):
                continue
            h = block.mlp.register_forward_pre_hook(make_hook(block))
            hooks.append(h)
        return hooks

    def _hook_mid_mlp_in_gpt(self):
        def make_hook(block):
            def hook(module, inputs, output):
                if getattr(block, "is_tailor_block", False):
                    return
                self.layer_representations.append(output)
            return hook

        hooks = []
        for block in self.encoder.h:
            if getattr(block, "is_tailor_block", False):
                continue
            h = block.mlp.c_fc.register_forward_hook(make_hook(block))
            hooks.append(h)
        return hooks

    def _hook_post_mlp_in_gpt(self):
        def make_hook(block):
            def hook(module, inputs, output):
                if getattr(block, "is_tailor_block", False):
                    return
                self.layer_representations.append(output)
            return hook

        hooks = []
        for block in self.encoder.h:
            if getattr(block, "is_tailor_block", False):
                continue
            h = block.mlp.c_proj.register_forward_hook(make_hook(block))
            hooks.append(h)
        return hooks

    def build_prefix_cache(self, batch_size, device, dtype):
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
        self.layer_attention_metrics["sum_attention_squared_to_each_layer"] += (
            attn_weights ** 2
        ).sum(dim=(0, 1))

        self.layer_attention_metrics["num_aggregation_layers"] = attn_weights.size(-1)
        self.layer_attention_metrics["num_tokens_aggregated"] += (
            attn_weights.size(0) * attn_weights.size(1)
        )

        avg_attention = (
            self.layer_attention_metrics["sum_attention_to_each_layer"]
            / self.layer_attention_metrics["num_tokens_aggregated"]
        )

        avg_attention_squared = (
            self.layer_attention_metrics["sum_attention_squared_to_each_layer"]
            / self.layer_attention_metrics["num_tokens_aggregated"]
        )

        std_attention = torch.sqrt(avg_attention_squared - avg_attention ** 2 + 1e-8)
        entropy_of_attention = -torch.sum(avg_attention * torch.log(avg_attention + 1e-8))

        self.layer_attention_metrics["avg_attention_to_each_layer"] = avg_attention
        self.layer_attention_metrics["std_attention_to_each_layer"] = std_attention
        self.layer_attention_metrics["entropy_of_layer_attention"] = entropy_of_attention

    def get_kl_loss(self):
        if self.variational_head is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self.variational_head.get_kl_loss()

    def get_delta_loss(self):
        if self.variational_head is None:
            raise RuntimeError("Variational modeling is disabled.")
        return self.variational_head.get_delta_loss()

    def get_variational_stats(self):
        if self.variational_head is None:
            raise RuntimeError("Variational modeling is disabled.")
        return self.variational_head.get_variational_stats()

    def forward_gpt(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
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

        self.layer_representations = []

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
                output_hidden_states=output_hidden_states,
                return_dict=return_dict if return_dict is not None else True,
                cache_position=cache_position,
            )

        expected_n_layers = self.config.n_layer
        got_n_layers = len(self.layer_representations)

        assert got_n_layers == expected_n_layers, (
            f"Expected {expected_n_layers} {self.representation_type} activations, "
            f"got {got_n_layers}"
        )

        reps = tuple(self.layer_representations)

        if self.num_aggregation_layers is not None:
            reps = reps[-self.num_aggregation_layers:]

        if attention_mask is not None:
            key_padding_mask = ~attention_mask[:, -encoder_outputs.last_hidden_state.size(1):].bool()
        else:
            key_padding_mask = None

        deterministic_delta, layer_token_attention = self.aggregator(
            base_hidden_state=encoder_outputs.last_hidden_state,
            layer_representations=reps,
            key_padding_mask=key_padding_mask,
        )

        if self.variational_head is not None:
            delta = self.variational_head(
                deterministic_delta,
                key_padding_mask=key_padding_mask,
            )
        else:
            delta = deterministic_delta

        aggregated_hidden_state = encoder_outputs.last_hidden_state + delta

        if key_padding_mask is not None:
            aggregated_hidden_state = aggregated_hidden_state.masked_fill(
                key_padding_mask.unsqueeze(-1),
                0.0,
            )

        self.layer_representations = []

        if layer_token_attention is not None:
            self.update_layer_attention_metrics(layer_token_attention)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=aggregated_hidden_state,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )