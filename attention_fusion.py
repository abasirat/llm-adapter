
import torch
from typing import List, Optional, Tuple, Union
from transformers import BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class AttentionFusion(torch.nn.Module):
    def __init__(self, encoder):
        super(AttentionFusion, self).__init__()

        if isinstance(encoder, str):
            import transformers
            encoder = transformers.AutoModel.from_pretrained(encoder)

        hidden_size = encoder.config.hidden_size
        num_hidden_layers = encoder.config.num_hidden_layers + 1

        # task specific attention query vector
        self.Q = torch.nn.Parameter(
                torch.ones(hidden_size,), 
                requires_grad=True
        )

        # the base transformer encoder
        self.encoder = encoder
 
        self.layernorm = torch.nn.LayerNorm([hidden_size,num_hidden_layers])
        self.dropout = torch.nn.Dropout(0.15)

        for param in self.encoder.parameters(): 
            param.requires_grad = False

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

    def forward(self, 
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

        # v: a four dimensional tensor of size BxTxDxL
        # B: batch size
        # T: sequence length
        # D: token vector dim
        # L: number of encoder layers
        v = torch.stack(encoder_outputs.hidden_states, -1)

        # recommended by Cao et al.
        v = self.layernorm(v)
        v = self.dropout(v)

        # alpha is a BxTxL tensor
        # alpha[b,i,j] is the attention weight of token i at layer j
        # Eq (1) in the Cao et. al. (2022) 
        alpha = v.transpose(2,3).matmul(self.Q)
        alpha = torch.nn.Softmax(-1)(alpha)

        # c is a tensor of size BxTxD holding the contextual representation
        # Eq (2) in the Cao et. al. (2022) 
        c = torch.einsum('BLT,BTLD->BTD', 
                alpha.transpose(1,2), 
                v.transpose(2,3)
        )

        c = torch.nn.GELU()(c)

        return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state = c,
                #pooler_output = None,
                #past_key_values=encoder_outputs.past_key_values,
                #hidden_states=encoder_outputs.hidden_states,
                #attentions=encoder_outputs.attentions,
                #cross_attentions=encoder_outputs.cross_attentions,
        )

class TokenClassifier(torch.nn.Module):
    def __init__(self, input_dim, out_dim):
        super(TokenClassifier, self).__init__()
        self.pip = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 256),
                torch.nn.Linear(256, out_dim)
        )
        
    def forward(self, x):
        return self.pip(x)
        

