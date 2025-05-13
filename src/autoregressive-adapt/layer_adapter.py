import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions

class LayerAdapter(torch.nn.Module):
    def __init__(self, 
            encoder, 
            hidden_size=None,
            num_heads=None
        ):
        super(LayerAdapter, self).__init__()

        input_size = encoder.config.hidden_size
        hs = hidden_size or input_size
        nh = num_heads or encoder.config.n_head

        self.encoder = encoder
        self.config = encoder.config

        self.quey_projection_layer = None
        self.key_projection_layer = None

        if hidden_size is not None:
            self.quey_projection_layer = torch.nn.Linear(input_size, hs)
            self.key_projection_layer = torch.nn.Linear(input_size, hs)
        
        self.token_layer_attention = torch.nn.MultiheadAttention(
                embed_dim = hs, 
                num_heads = nh, 
                kdim = hs,
                vdim = hs,
                batch_first = True
        )

        for param in self.encoder.parameters(): 
            param.requires_grad = False
        
        if self.encoder.name_or_path.startswith("gpt"):
            self.forward = self.forward_gpt
        elif self.encoder.name_or_path.startswith("bert"):
            self.forward = self.forward_bert
        else:
            print("this implementation works only with standard gpt and bert families from huggingface")
            raise NotImplementedError
        
        #self.print_trainable_parameters()

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

    def aggregator(self, inputs, key_padding_mask):
        Q = torch.mean(inputs,-1)
        if self.quey_projection_layer:
            Q = self.quey_projection_layer(Q)

        K = torch.mean(inputs,1).transpose(1,2)
        if self.key_projection_layer:
            K = self.key_projection_layer(K)

        _, NQK = self.token_layer_attention(Q,K,K,need_weights=True)
        if key_padding_mask is not None:
            NQK[key_padding_mask] = 0 
        r = torch.einsum('bdnm,bnm->bnd', inputs.transpose(1,2), NQK)
        return r, NQK

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
        )-> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

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
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
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
        )

        # do not pad for new generated input if only decoder
        if self.encoder.config.is_encoder_decoder is False:
            key_padding_mask = None
        else:
            key_padding_mask = torch.logical_not(attention_mask)
        hs = torch.stack(encoder_outputs.hidden_states, -1) 
        aggregated_hidden_state, layer_token_attention = self.aggregator(hs, key_padding_mask)

        return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=aggregated_hidden_state,
                    past_key_values=encoder_outputs.past_key_values,
                    hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
                    attentions=encoder_outputs.attentions,
                    cross_attentions=encoder_outputs.cross_attentions,
                )