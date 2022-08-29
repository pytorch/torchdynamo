import os

import torch
from datasets import load_dataset
from datasets import load_metric
from torch.optim import SGD
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import torchdynamo

torchdynamo.config.fake_tensor_propagation = False

# You will download around 84G dataset if you run this end to end training/evaluation example.

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


def training_iter_fn(batch, model, optimizer):
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


def model_training_evaluation(
    train_dataloader, eval_dataloader, model, optimizer, num_epochs
):
    model.to(device)
    model.train()
    loss_history = []
    # Support backends: eager, aot_nop and aot_nvfuser
    opt_training_iter_fn = torchdynamo.optimize("eager")(training_iter_fn)
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
    opt_model = torchdynamo.optimize("eager")(model)
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = opt_model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print("Loss history : " + str(loss_history))
    print("Accuracy : " + str(metric.compute()))


if __name__ == "__main__":
    train_dataloader, eval_dataloader = data_processing(num_samples)
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )
    optimizer = SGD(model.parameters(), lr=5e-5, momentum=0.9)
    model_training_evaluation(
        train_dataloader, eval_dataloader, model, optimizer, num_epochs
    )
