import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions

class Tailor(torch.nn.Module):
    def __init__(self, input_size:int, dropout:float, hdim:int, num_heads:int, tailor_attention:bool):
        super(Tailor, self).__init__()
        self.FF = torch.nn.Sequential(
                torch.nn.LayerNorm(input_size, eps=1e-12),
                torch.nn.Linear(input_size, input_size),
                torch.nn.Dropout(dropout),
                torch.nn.ReLU()
                )

        self.tailor_attention = tailor_attention

        if tailor_attention:
            #self.tailor_key_dim_reducer = torch.nn.Linear(input_size, ff_dim)
            self.tailor_query_dim_reducer = torch.nn.Linear(input_size, hdim)
            self.self_attention = torch.nn.MultiheadAttention(
                    embed_dim = hdim,
                    num_heads = num_heads,
                    kdim = hdim,
                    vdim = hdim,
                    batch_first = True)
            self.att_dropout = torch.nn.Dropout(dropout)

            torch.nn.init.xavier_uniform_(self.tailor_query_dim_reducer.weight)
            torch.nn.init.xavier_uniform_(self.self_attention.in_proj_weight)
            torch.nn.init.xavier_uniform_(self.self_attention.out_proj.weight)

    def forward(self, inputs, key_padding_mask):
        x = inputs + self.FF(inputs.clone())

        if self.tailor_attention:
            x = self.att_dropout(x)

            Q = self.tailor_query_dim_reducer(x)
            #K = self.tailor_key_dim_reducer(x)
            _, NQK = self.self_attention(Q,Q,Q,
                    key_padding_mask=key_padding_mask,
                    need_weights=True)
            x = NQK.bmm(x)

        return x


class Adapter(torch.nn.Module):
    def __init__(self, 
            encoder, 
            aggregation_layers=None,
            num_heads=2,
            hidden_size=16,
            dropout=0.1,
            enable_tailor=True,
            tailor_attention=True,
        ):
        super(Adapter, self).__init__()

        if isinstance(encoder, str):
            import transformers
            encoder = transformers.AutoModel.from_pretrained(encoder)

        input_size = encoder.config.hidden_size
        num_hidden_layers = encoder.config.num_hidden_layers + 1

        self.encoder = encoder

        if aggregation_layers is None: 
            aggregation_layers = [True]*num_hidden_layers
        else:
            # aggregation_layers = [l in aggregation_layers for l in range(num_hidden_layers)]
            print("this implementation works only on all layers")
            raise NotImplementedError


        assert len(aggregation_layers) == num_hidden_layers
        self.aggregation_layers = torch.tensor(aggregation_layers, dtype=torch.bool)

        self.key_projector = torch.nn.Linear(input_size, hidden_size)
        self.query_projector = torch.nn.Linear(input_size, hidden_size)
        self.token_layer_attention = torch.nn.MultiheadAttention(
                embed_dim = hidden_size, 
                num_heads = num_heads, 
                kdim = hidden_size,
                vdim = hidden_size,
                batch_first = True
        )

        torch.nn.init.xavier_uniform_(self.token_layer_attention.in_proj_weight)
        torch.nn.init.xavier_uniform_(self.token_layer_attention.out_proj.weight)
        #self.token_layer_attention.out_proj.bias.fill_(0.0)

        if enable_tailor:
            self.tailor = Tailor(input_size, dropout, hidden_size, num_heads, tailor_attention)
        else:
            self.tailor = None

        for param in self.encoder.parameters(): 
            param.requires_grad = False
        
        if self.encoder.name_or_path.startswith("gpt"):
            self.forward = self.forward_gpt
        elif self.encoder.name_or_path.startswith("bert"):
            self.forward = self.forward_bert
        else:
            print("this implementation works only with standard gpt and bert families from huggingface")
            raise NotImplementedError
        
        self.print_trainable_parameters()

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
        print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")



    def aggregator(self, inputs, key_padding_mask):
        Q = torch.mean(inputs,-1)
        Q = self.query_projector(Q)
        K = torch.mean(inputs,1).transpose(1,2)
        K = self.key_projector(K)
        _, NQK = self.token_layer_attention(Q,K,K,need_weights=True)
        NQK[key_padding_mask] = 0 
        r = torch.einsum('bdnm,bnm->bnd', inputs.transpose(1,2), NQK)
        return r, NQK

    def adapter(self, inputs, key_padding_mask=None):
        # inputs = inputs[..., self.aggregation_layers]
        r, att = self.aggregator(inputs, key_padding_mask) 
        if self.tailor:
            r += self.tailor(r, key_padding_mask)
        return r, att

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
        #hs = torch.stack(encoder_outputs[2], -1)
        hs = torch.stack(encoder_outputs.hidden_states, -1)
        #pdb.set_trace()
        aggregated_hidden_state, layer_token_attention = self.adapter(hs, key_padding_mask)

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
        
        key_padding_mask = torch.logical_not(attention_mask)
        #hs = torch.stack(encoder_outputs[2], -1)
        hs = torch.stack(encoder_outputs.hidden_states, -1)
        aggregated_hidden_state, layer_token_attention = self.adapter(hs, key_padding_mask)

        return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=aggregated_hidden_state,
                    past_key_values=encoder_outputs.past_key_values,
                    hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
                    attentions=encoder_outputs.attentions,
                    cross_attentions=encoder_outputs.cross_attentions,
                )