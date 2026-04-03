import torch
from typing import Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

class LanguageAdapter(torch.nn.Module):
    def __init__(self, 
            encoder, 
            num_tailor_layers=1,
            dropout=0.1,
        ):
        super(LanguageAdapter, self).__init__()

        self.encoder = encoder
        self.config = self.encoder.config

        self.num_tailor_layers = num_tailor_layers
        self.tailor_blocks = torch.nn.ModuleList(
            [GPT2Block(encoder.config, layer_idx=i, dropout=dropout) for i in range(num_tailor_layers)]
        )
        
        device = next(self.parameters()).device
        ctx_len = encoder.config.n_ctx
        self.causal_mask = torch.triu(torch.ones(ctx_len, ctx_len, device=device), diagonal=1).bool()
        
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
        print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")
    
    def tailor_old(self, hidden_states):
        seq_len = hidden_states.size()[1]
        #causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1).bool()
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        for block in self.tailor_blocks:
            hidden_states = block(hidden_states, attention_mask=causal_mask)

            if type(hidden_states) == tuple:
                hidden_states = hidden_states[0]

        return hidden_states
    
    def tailor(self, hidden_states, attention_mask=None):
        for block in self.tailor_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        return hidden_states

    def forward(
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
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
        )

        #hs = self.tailor(encoder_outputs.last_hidden_state)
        tailor_attention_mask = None
        if attention_mask is not None:
            # input attention_mask is usually (B, T) with 1 for tokens, 0 for padding
            tailor_attention_mask = attention_mask.to(dtype=encoder_outputs.last_hidden_state.dtype)
            tailor_attention_mask = tailor_attention_mask[:, None, None, :]  # (B, 1, 1, T)
            tailor_attention_mask = (1.0 - tailor_attention_mask) * torch.finfo(encoder_outputs.last_hidden_state.dtype).min

        hs = self.tailor(
            encoder_outputs.last_hidden_state,
            attention_mask=tailor_attention_mask,
        )


        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hs,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )