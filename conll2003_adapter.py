# based on https://huggingface.co/learn/nlp-course/chapter7/2

from datasets import load_dataset
from transformers import GPT2TokenizerFast, BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
import evaluate
import numpy as np
import torch
from attention_fusion import TokenClassifier
from adapter import Adapter
import math

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            #new_labels.append(label)
            new_labels.append(-100)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
                examples["tokens"], 
                truncation=True, 
                is_split_into_words=True)
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
           [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
           for prediction, label in zip(predictions, labels)]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
    }

def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # np.random.seed(seed)
    
if __name__ == '__main__':
    if torch.cuda.is_available(): 
        device = 'cuda'
    elif torch.backends.mps.is_available(): 
        device = 'mps'
    else: 
        device = 'cpu'
    device = torch.device(device)
    print(f"working device: {device}")

    set_seed(1)

    raw_datasets = load_dataset("conll2003")
    ner_feature = raw_datasets["train"].features["ner_tags"]
    label_names = ner_feature.feature.names

    model_checkpoint = "bert-large-cased"
    #model_checkpoint = "gpt2"
    if model_checkpoint.startswith("bert"):
        tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, add_prefix_space=True)
    elif model_checkpoint.startswith("gpt"):
        tokenizer = GPT2TokenizerFast.from_pretrained(model_checkpoint, add_prefix_space=True)
    
    tokenizer.pad_token = '[PAD]'
    tokenizer.eos_token = '[PAD]'

    tokenized_datasets = raw_datasets.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            id2label=id2label,
            label2id=label2id,
    ).to(device)

    metric = evaluate.load("seqeval")

    args = TrainingArguments(
        f"{model_checkpoint}-ner",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=1e-3,
        num_train_epochs=10,
        weight_decay=0.01,
        push_to_hub=False,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        dataloader_num_workers=3,
    )

    adapter_params = {'enable_tailor':False, 'num_heads':8, 'hidden_size':16, 'tailor_attention':True}
    if model_checkpoint.startswith("bert"):
        model.bert = Adapter(model.bert, **adapter_params).to(device)
    elif model_checkpoint.startswith("gpt"):
        model.transformer = Adapter(model.transformer, **adapter_params).to(device)
    
    model.classifier = TokenClassifier(
            model.config.hidden_size, 
            len(label2id)
    )
      
    num_training_batches = math.ceil(len(tokenized_datasets["train"]) / args.per_device_train_batch_size)  
    num_training_steps = num_training_batches * args.num_train_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate, num_training_steps)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler),
    )
    trainer.train()

    test_dataset = tokenized_datasets["test"]
    test_eval = trainer.evaluate(test_dataset)
    print(test_eval)

    with open('adapter_train_history.log', 'w') as fp:
        print(trainer.state.log_history, file=fp)


