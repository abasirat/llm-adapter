import pdb

import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions

class LayerAdapter(torch.nn.Module):
    def __init__(self, 
            encoder, 
            hidden_size=None,
            num_heads=None,
            need_weights=False,
            dropout=0.1,
        ):
        super(LayerAdapter, self).__init__()

        input_size = encoder.config.hidden_size
        hs = hidden_size or input_size
        nh = num_heads or encoder.config.n_head
        nl = encoder.config.n_layer + 1

        self.encoder = encoder
        self.config = encoder.config

        self.encoder.eval() # Set encoder to evaluation mode to disable dropout and other training-specific layers. It will speed up inference and reduce memory usage. We will enable grad only for the adapters and lm_head.

        self.query_projection_layer = None
        self.key_projection_layer = None
        self.need_weights = need_weights

        if hidden_size is not None:
            self.query_projection_layer = torch.nn.Linear(input_size, hs)
            self.key_projection_layer = torch.nn.Linear(input_size, hs)
        
        self.token_layer_attention = torch.nn.MultiheadAttention(
                embed_dim = hs, 
                num_heads = nh, 
                kdim = hs,
                vdim = hs,
                batch_first = True,
                dropout = dropout,
        )

        self.inear_transforms = torch.nn.ModuleDict({
            f"layer_{i}": torch.nn.Linear(input_size, hs) for i in range(nl)
        })

        for param in self.encoder.parameters(): 
            param.requires_grad = False
        
        if self.encoder.name_or_path.startswith("gpt"):
            self.forward = self.forward_gpt
        elif self.encoder.name_or_path.startswith("bert"):
            self.forward = self.forward_bert
        else:
            print("this implementation works only with standard gpt and bert families from huggingface")
            raise NotImplementedError
        
        self.output_dropout = torch.nn.Dropout(dropout)

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

    def aggregator_old(self, inputs, key_padding_mask):
        Q = torch.mean(inputs,-1)
        if self.query_projection_layer:
            Q = self.query_projection_layer(Q)

        K = torch.mean(inputs,1).transpose(1,2)
        if self.key_projection_layer:
            K = self.key_projection_layer(K)

        _, NQK = self.token_layer_attention(Q,K,K,need_weights=True)
        if key_padding_mask is not None:
            NQK[key_padding_mask] = 0 
        r = torch.einsum('bdnm,bnm->bnd', inputs.transpose(1,2), NQK)
        return r, NQK
    
    def aggregator(self, inputs, key_padding_mask=None, need_weights=False):
        """
        inputs: (B, T, D, L)
            B = batch
            T = sequence length
            D = hidden size
            L = number of hidden-state layers being aggregated

        Returns:
            r:   (B, T, D)      aggregated representation
            w:   (B, T, L)      per-token layer weights
        """

        # Query from the same token position, averaged over layers
        # Q: (B, T, D)
        #Q = inputs.mean(dim=-1)
        Q = inputs[..., -1] # use the top layer as query
        if self.query_projection_layer is not None:
            Q = self.query_projection_layer(Q)

        # Keys/values are the layer representations for the SAME token only.
        # Rearrange to: (B*T, L, D)
        KV = inputs.permute(0, 1, 3, 2).contiguous().view(-1, inputs.size(-1), inputs.size(2))

        K = KV
        if self.key_projection_layer is not None:
            K = self.key_projection_layer(K)
        #V = KV

        # Apply separate linear layer to each layer's representation before attention
        V = torch.stack([self.inear_transforms[f"layer_{i}"](inputs[..., i]) for i in range(inputs.size(-1))], dim=-1).permute(0, 1, 3, 2).contiguous().view(-1, inputs.size(-1), inputs.size(2))

        # Query per token: (B*T, 1, D)
        Q_bt = Q.contiguous().view(-1, 1, Q.size(-1))

        # Attend from each token to its own layer stack only.
        # No token-token mixing => causal by construction.
        attn_output, attn_weights = self.token_layer_attention(
            Q_bt, K, V, need_weights=need_weights
        )

        r = attn_output.squeeze(1).view(inputs.size(0), inputs.size(1), -1)
        if attn_weights is not None:
            attn_weights = attn_weights.squeeze(1).view(inputs.size(0), inputs.size(1), inputs.size(-1))

        if key_padding_mask is not None:
            r = r.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            if attn_weights is not None:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        
        r = self.output_dropout(r)

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
        hs = torch.stack(encoder_outputs.hidden_states, -1)
        aggregated_hidden_state, layer_token_attention = self.aggregator(hs, key_padding_mask)

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

        hs = torch.stack(encoder_outputs.hidden_states, -1) 
        aggregated_hidden_state, layer_token_attention = self.aggregator(hs, key_padding_mask, self.need_weights)

        return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=aggregated_hidden_state,
                    past_key_values=encoder_outputs.past_key_values,
                    hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
                    attentions=encoder_outputs.attentions,
                    cross_attentions=encoder_outputs.cross_attentions,
                )