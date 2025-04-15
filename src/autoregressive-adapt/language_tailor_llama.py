import torch
from typing import Optional, Tuple, Union
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

# Works for transformers==4.45.2

class LanguageAdapter(torch.nn.Module):
    def __init__(self, 
            encoder, 
            num_tailor_layers=1,
        ):
        super(LanguageAdapter, self).__init__()

        self.encoder = encoder
        self.config = self.encoder.config
        self.num_tailor_layers = num_tailor_layers

        self.tailor_blocks = torch.nn.ModuleList(
            [LlamaDecoderLayer(self.config, layer_idx=i) for i in range(num_tailor_layers)]
        )
#         self.rotary_emb = LlamaRotaryEmbedding(config=encoder.config)  # uncomment for using transformers>=4.46
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
    

    def tailor(self, hidden_states):
        batch_size, seq_len = hidden_states.size()[:2]

        # Create causal mask        
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == float('-inf'), -1e9)  

        # Then convert to 4D: [batch_size, 1, seq_len, seq_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len)
        
        # Create position ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        
        # Get rotary embeddings using the imported class
#         position_embeddings = self.rotary_emb(hidden_states, position_ids)  # uncomment for using transformers>=4.46
        
        for block in self.tailor_blocks:
            hidden_states = block(
                hidden_states, 
                attention_mask=causal_mask,
                position_ids=position_ids,
#                 position_embeddings=position_embeddings  # uncomment for using transformers>=4.46
            )[0]
        
        return hidden_states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[int] = None, 
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
        encoder_outputs =  self.encoder(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,  
        )

        hs = self.tailor(encoder_outputs.last_hidden_state)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hs,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )