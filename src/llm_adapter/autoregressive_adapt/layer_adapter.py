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
class LayerAdapter(torch.nn.Module):
    def __init__(self, 
            encoder, 
            need_weights=False,
            dropout=0.1,
            num_aggregation_layers=None,
        ):
        super(LayerAdapter, self).__init__()

        hs = encoder.config.hidden_size
        nh = encoder.config.n_head
        nl = encoder.config.n_layer

        self.encoder = encoder
        self.config = encoder.config

        self.encoder.eval() # Set encoder to evaluation mode to disable dropout and other training-specific layers. It will speed up inference and reduce memory usage. We will enable grad only for the adapters and lm_head.

        self.need_weights = need_weights
        self.num_aggregation_layers = num_aggregation_layers

        self.token_layer_attention = torch.nn.MultiheadAttention(
                embed_dim = hs, 
                num_heads = nh, 
                kdim = hs,
                vdim = hs,
                batch_first = True,
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
        Q = hidden_states[..., -1] # use the top layer as query
        # Keys/values are the layer representations for the SAME token only.
        # Rearrange to: (B*T, ??
        KV = inputs.permute(0, 1, 3, 2).contiguous().view(-1, inputs.size(-1), inputs.size(2))

        K = KV
        V = KV

        # Query per token: (B*T, 1, D)
        Q_bt = Q.contiguous().view(-1, 1, Q.size(-1))

        # Attend from each token to its own layer stack only.
        # No token-token mixing => causal by construction.
        attn_output, attn_weights = self.token_layer_attention(
            Q_bt, K, V , need_weights=self.need_weights
        )

        #r = attn_output.squeeze(1).view(inputs.size(0), inputs.size(1), -1)
        base = hidden_states[..., -1]
        delta = attn_output.squeeze(1).view(inputs.size(0), inputs.size(1), -1)
        r = base + self.adapter_scale * self.output_dropout(delta)

        if attn_weights is not None:
            attn_weights = attn_weights.squeeze(1).view(inputs.size(0), inputs.size(1), inputs.size(-1))

        if key_padding_mask is not None:
            r = r.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            if attn_weights is not None:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        
        #r = self.output_dropout(r)

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

        return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state = aggregated_hidden_state,
                pooler_output = encoder_outputs.pooler_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
                attentions=encoder_outputs.attentions,
                cross_attentions=encoder_outputs.cross_attentions,
        )

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
        
        with torch.no_grad():
            encoder_outputs =  self.encoder(
                    input_ids,
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

        ## do not pad for new generated input if only decoder
        #if self.encoder.config.is_encoder_decoder is False:
        #    key_padding_mask = None
        #else:
        #    key_padding_mask = torch.logical_not(attention_mask)

        # Keep padding mask when present, even for GPT-style models.
        #if attention_mask is not None:
        #    key_padding_mask = torch.logical_not(attention_mask.bool())
        #else:
        #    key_padding_mask = None

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        assert len(self.before_mlp_activations) == self.config.n_layer, \
            f"Expected {self.config.n_layer} pre-MLP activations, got {len(self.before_mlp_activations)}"
        
        hs = torch.stack(encoder_outputs.hidden_states, -1)[..., 1:] # shape (B, T, D, L)
        ac = torch.stack(self.before_mlp_activations, -1) #  shape (B, T, D, L)
        if self.num_aggregation_layers is not None:
            hs = hs[..., -self.num_aggregation_layers:]
            ac = ac[..., -self.num_aggregation_layers:]
        aggregated_hidden_state, layer_token_attention = self.aggregator(hs, ac, key_padding_mask)
        self.before_mlp_activations = [] # clear stored activations after use

        return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=aggregated_hidden_state,
                    past_key_values=encoder_outputs.past_key_values,
                    hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
                    attentions=encoder_outputs.attentions,
                    cross_attentions=encoder_outputs.cross_attentions,
                )