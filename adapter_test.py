import torch
import transformers
from attention_fusion import AttentionFusion
from adapter import Adapter

import unittest

class Encoder(torch.nn.Module):
    def __init__(self, tokenizer, adapter):
        super(Encoder, self).__init__()

        self.tokenizer = tokenizer
        self.adapter = adapter.to(device)

    def forward(self, inputs):
        x = self.tokenizer(inputs[0], inputs[1], 
            is_split_into_words=True, 
            return_tensors="pt",
            padding="longest", 
            truncation=True).to(device)

        return self.adapter(**x)

class EncoderTest(unittest.TestCase):

    def test_attention_fusion(self):
        print("Attention Fusion")
        adapter = AttentionFusion(bert)
        adapter.print_trainable_parameters()
        encoder = Encoder(tokenizer, adapter)
        y = encoder(text)
        print(y.last_hidden_state.shape)

    def test_adapter(self):
        print("The Adapter Model")
        adapter = Adapter(bert)
        adapter.print_trainable_parameters()
        encoder = Encoder(tokenizer, adapter)
        y = encoder(text)
        print(y.last_hidden_state.shape)

if __name__ == '__main__':
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    bertstr = 'bert-base-cased'

    tokenizer = transformers.AutoTokenizer.from_pretrained(bertstr) 
    bert = transformers.AutoModel.from_pretrained(bertstr)
    
    text = [
            [
                ["the cat sat on the mat ."],
                ["boys and girls fool each other ."],
                ["we do not feel some others' feelings unless we have experienced their feelings."]
            ],
                None
           ]

    unittest.main()

