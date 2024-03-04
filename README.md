# Efficient Adaptation of Large Language Models

Transformer-based encoders, such as large language models, are becoming one of the fundamental components of deep learning models. Although they deliver exceptional performance, one of the main issues with using such encoders (e.g., BERT and GPT) in downstream tasks is their high demand for computational resources, such as memory and processing units. To address this problem, we provide a PyTorch implementation of an adapter model that enables efficient adaptation of such encoders to downstream tasks. We also offer detailed information about the adapter's functionality, performance, and efficiency in the following paper:

> Ali Basirat, 2024, **Efficient Structured Prediction with Transformer Encoders**, *Northern European Journal of Language Technology* (NEJLT), In Press.


In the example below, we wrap a BERT encoder with an `Adapter` module.
```python
import transformers
from adapter import Adapter

bert = transformers.AutoModel.from_pretrained('bert-large-cased')
adapter = Adapter(bert)
```
The object `adapter` includes the BERT model plus an adapter block on top of it. Note that the `adapter` freezes the BERT model during training (i.e., only the added parameters are trained). 

As an example, we have included a sample code in `conll2003_adapter.py` that demonstrates how to integrate the `adapter` into a BERT or GPT model for named entity recognition (NER). These implementations are built on the standard huggingface models but can be expanded to other Transformer-based encoders. The code results in an $F_1$-score of $89.3$ and $83.1$ for named entity recognition on the CoNLL-2003 data set with `bert-large-cased` and `gpt2` models, respectively.  

If you utilize this software for research purposes, please include the following citation:

    @article{XX,
        author    = {Ali Basirat},
        title     = {Efficient Structured Prediction with Transformer Encoders},
        journal   = {Northern European Journal of Language Technology ({NEJLT})},
        volume    = {X},
        pages     = {xx--xx},
        year      = {2024},
        url       = {https://xx.com},
        biburl    = {xx},
        bibsource = {xx}
    }


## Practical hints

- For further memory efficiency gain, you can disable the adaptor block whose contribution is more apparent to the document classification tasks rather than the structured prediction. For detailed analysis of the tailor block see Section 6 of the paper.
```python
adapter = Adapter(bert, enable_tailor=False)
```
- Saving the encoder activation once and reusing it while training is recommended to obtain a high-efficiency gain during training. See section 5.3 for further detailed information on this. 