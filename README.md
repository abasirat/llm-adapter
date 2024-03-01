Transformer-based encoders, such as large language models, are becoming one of the fundamental components of deep learning models. Although they deliver exceptional performance, one of the main issues with using such encoders (e.g., BERT and GPT) in downstream tasks is their high demand for computational resources, such as memory and processing units, to adapt them for the tasks. To address this problem, we provide a PyTorch implementation of an adapter model that enables efficient adaptation of such encoders. We also offer detailed information about the adapter's functionality, performance, and efficiency in the following paper:

Ali Basirat, 2024, Efficient Structured Prediction with Transformer, Northern European Journal for Language Technology (NEJLT).

In the example below, we wrap a BERT encoder with an `Adapter` module.
```python
import torch, transformers
from adapter import Adapter

bert = transformers.AutoModel.from_pretrained('bert-large-cased')
adapter = Adapter(bert)
```
The object `adapter` includes the BERT model plus an adapter block on top of it. 

