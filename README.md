# Efficient Adaptation of Large Language Models

Transformer-based encoders, such as large language models, are becoming one of the fundamental components of deep learning models. Although they deliver exceptional performance, one of the main issues with using such encoders (e.g., BERT and GPT) in downstream tasks is their high demand for computational resources, such as memory and processing units. To address this problem, we provide a PyTorch implementation of an adapter model that enables efficient adaptation of such encoders to downstream tasks. We also offer detailed information about the adapter's functionality, performance, and efficiency in the following paper:

> [Ali Basirat, 2024, **Efficient Structured Prediction with Transformer Encoders**, *The Northern European Journal of Language Technology* (NEJLT), 10(1), Linköping University Electronic Press](https://nejlt.ep.liu.se/article/view/4932).


In the example below, we wrap a BERT encoder with an `Adapter` module.
```python
import transformers
from adapter import Adapter

bert = transformers.AutoModel.from_pretrained('bert-large-cased')
adapter = Adapter(bert)
```
The object `adapter` includes the BERT model plus an adapter block on top of it. Note that the `adapter` freezes the BERT model during training (i.e., only the added parameters are trained). 

As an example, we have included a sample code in `conll2003_adapter.py` that demonstrates how to integrate the `adapter` into a BERT or GPT model for named entity recognition (NER). These implementations are built on the standard huggingface models but can be extended to other Transformer-based encoders. The code's performance for named entity recognition on the CoNLL-2003 data set with a fixed seed value is as below: 

Model | F1-score 
--- | ---
bert-base-cased | 88.8 
bert-large-cased | 89.3 
roberta-base | 89.3
roberta-large | 89.8
gpt2 | 83.1 
gpt2-medium | 81.1 

If you utilize this software for research purposes, please include the following citation:

    @article{basirat2024adapter,
        author    = {Ali Basirat},
        title     = {Efficient Structured Prediction with Transformer Encoders},
        journal   = {The Northern European Journal of Language Technology ({NEJLT})},
        volume    = {10},
        number    = {1},
        pages     = {1--13},
        year      = {2024},
        url       = {https://nejlt.ep.liu.se/article/view/4932},
    }


## Practical hints

- For further memory efficiency gain, you can disable the adaptor block whose contribution is more apparent to the document classification tasks rather than the structured prediction. For a detailed analysis of the tailor block see Section 6 of the paper.
```python
adapter = Adapter(bert, enable_tailor=False)
```
- Saving the encoder activation once and reusing it while training is recommended to obtain a high-efficiency gain during training. See Section 5.3 of the paper for further detailed information on this. 