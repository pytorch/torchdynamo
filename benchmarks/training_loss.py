import os

import torch
from datasets import load_dataset
from datasets import load_metric
from torch.optim import SGD
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import torchdynamo

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_samples = 10000
num_epochs = 3


def data_processing(num_samples):
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

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    return train_dataloader, eval_dataloader


@torchdynamo.optimize("eager")
def training_iter_fn(batch, model, optimizer):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


@torchdynamo.optimize("eager")
def eval_iter_fn(batch):
    return model(**batch)


def model_training_evaluation(
    train_dataloader, eval_dataloader, model, optimizer, num_epochs
):
    model.to(device)
    model.train()
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader, 0):
            loss = training_iter_fn(batch, model, optimizer)
            running_loss += loss.item()
            if i % 100 == 99:
                loss_history.append(running_loss / 100)
                running_loss = 0.0

    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = eval_iter_fn(batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    return loss_history, metric.compute()


if __name__ == "__main__":
    train_dataloader, eval_dataloader = data_processing(num_samples)
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )
    optimizer = SGD(model.parameters(), lr=5e-5, momentum=0.9)
    loss_history, accuracy = model_training_evaluation(
        train_dataloader, eval_dataloader, model, optimizer, num_epochs
    )
    print("Loss history : " + str(loss_history))
    print("Accuracy : " + str(accuracy))
