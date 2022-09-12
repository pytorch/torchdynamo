import argparse
import os
import sys

import torch
from datasets import load_dataset
from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import torchdynamo

torchdynamo.config.fake_tensor_propagation = False

# You will download around 84G dataset if you run this end to end training/evaluation example.

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def data_processing(num_samples, batch_size):
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(num_samples))
    )
    small_eval_dataset = (
        tokenized_datasets["test"].shuffle(seed=42).select(range(num_samples))
    )

    train_dataloader = DataLoader(
        small_train_dataset, shuffle=True, batch_size=batch_size
    )
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader


def training_iter_fn(batch, model, optimizer):
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


def model_training_evaluation(
    backend, train_dataloader, eval_dataloader, model, optimizer, num_epochs
):
    model.to(device)
    model.train()
    loss_history = []
    # Support backends: eager, aot_nop and aot_nvfuser
    opt_training_iter_fn = torchdynamo.optimize(backend)(training_iter_fn)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader, 0):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = opt_training_iter_fn(batch, model, optimizer)
            running_loss += loss.item()
            if i % 100 == 99:
                loss_history.append(running_loss / 100)
                running_loss = 0.0

    metric = load_metric("accuracy")
    model.eval()
    opt_model = torchdynamo.optimize(backend)(model)
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = opt_model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print("Loss history : " + str(loss_history))
    print("Accuracy : " + str(metric.compute()))


def parse_args():
    parser = argparse.ArgumentParser(
        description="TorchDynamo end to end training/evaluation benchmark"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="number of samples to train/eval (default: 10000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--backend",
        choices=torchdynamo.list_backends(),
        default="eager",
        help="train/evaluate model with a given backend (default: eager)",
    )
    parser.add_argument(
        "--optimizer",
        default="SGD",
        help="train model using a given optimizer (default: SGD)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train_dataloader, eval_dataloader = data_processing(
        args.num_samples, args.batch_size
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )
    optimizer_cls = getattr(sys.modules["torch.optim"], args.optimizer)
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    model_training_evaluation(
        args.backend, train_dataloader, eval_dataloader, model, optimizer, args.epochs
    )
