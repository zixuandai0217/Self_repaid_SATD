from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from temporal_utils import DEFAULT_FIXED_TEST_TRAIN_PERCENTAGES, make_fixed_test_temporal_splits


PLAN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PLAN_DIR / "data" / "merged_data_40501_temporal_sorted_by_add_date.json"
DEFAULT_METRICS_CSV = PLAN_DIR / "results" / "temporal_text_model_metrics.csv"
DEFAULT_SUMMARY_MD = PLAN_DIR / "results" / "temporal_text_model_summary.md"

PAD_INDEX = 0
UNKNOWN_INDEX = 1


@dataclass(frozen=True)
class TextCNNConfig:
    """Store TextCNN hyperparameters from the original predictive_models script."""

    max_words: int = 20000
    max_len: int = 100
    embedding_dim: int = 300
    filter_sizes: tuple[int, ...] = (1, 2, 3, 4, 5)
    num_filters: int = 200
    dropout_rate: float = 0.5
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 50
    patience: int = 3


@dataclass(frozen=True)
class BertConfig:
    """Store BERT hyperparameters from the original predictive_models script."""

    model_name: str = "bert-base-uncased"
    max_len: int = 128
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 2e-5
    embed_dim: int = 300
    dropout_rate: float = 0.3
    patience: int = 3


TEXTCNN_CONFIG = TextCNNConfig()
BERT_CONFIG = BertConfig()
DEFAULT_PROTOCOL = "remaining-test"


def parse_percentages(text: str) -> list[int]:
    """Parse comma-separated train percentages."""

    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one train percentage is required")
    return values


def parse_models(text: str) -> list[str]:
    """Parse comma-separated text model names."""

    models = [part.strip().lower() for part in text.split(",") if part.strip()]
    unsupported = sorted(set(models) - {"textcnn", "bert"})
    if unsupported:
        raise ValueError(f"Unsupported text models: {', '.join(unsupported)}")
    return models


def load_records(path: Path) -> list[dict]:
    """Load the temporal SATD JSON dataset."""

    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {path}")
    return records


def make_split_indices(n_records: int, train_percentage: int) -> tuple[list[int], list[int]]:
    """Create chronological train/test indices for one temporal split."""

    if train_percentage <= 0 or train_percentage >= 100:
        raise ValueError("train_percentage must be between 1 and 99")
    train_size = int(n_records * train_percentage / 100)
    if train_size <= 0 or train_size >= n_records:
        raise ValueError(f"Invalid train size {train_size} for {n_records} records")
    return list(range(train_size)), list(range(train_size, n_records))


def apply_sample_limit(indices: list[int], limit: int | None) -> list[int]:
    """Keep the latest indices when a quick trial limit is requested."""

    if limit is None or limit <= 0 or len(indices) <= limit:
        return indices
    return indices[-limit:]


def train_validation_split(indices: list[int], validation_fraction: float) -> tuple[list[int], list[int]]:
    """Split the training window into chronological train and validation portions."""

    if validation_fraction <= 0:
        return indices, indices
    validation_size = max(1, int(len(indices) * validation_fraction))
    if validation_size >= len(indices):
        validation_size = max(1, len(indices) // 5)
    return indices[:-validation_size], indices[-validation_size:]


def make_text_protocol_indices(
    n_records: int,
    train_percentage: int,
    protocol: str,
    validation_fraction: float = 0.1,
    validation_start_percentage: int = 80,
    test_start_percentage: int = 90,
) -> tuple[list[int], list[int], list[int]]:
    """Create train, validation, and test indices for one text-model temporal split."""

    if protocol == "remaining-test":
        full_train_indices, test_indices = make_split_indices(n_records, train_percentage)
        train_indices, validation_indices = train_validation_split(
            full_train_indices, validation_fraction=validation_fraction
        )
        return train_indices, validation_indices, test_indices

    if protocol == "fixed-test":
        if not (0 < validation_start_percentage < test_start_percentage < 100):
            raise ValueError("Expected 0 < validation_start_percentage < test_start_percentage < 100")
        if train_percentage <= 0 or train_percentage > validation_start_percentage:
            raise ValueError(
                "train_percentage must be between 1 and the validation start percentage "
                f"({validation_start_percentage})"
            )
        train_size = int(n_records * train_percentage / 100)
        validation_start = int(n_records * validation_start_percentage / 100)
        test_start = int(n_records * test_start_percentage / 100)
        if train_size <= 0 or train_size > validation_start or test_start >= n_records:
            raise ValueError(f"Invalid fixed-test split for {n_records} records")
        return (
            list(range(train_size)),
            list(range(validation_start, test_start)),
            list(range(test_start, n_records)),
        )

    raise ValueError(f"Unsupported temporal protocol: {protocol}")


def make_text_protocol_indices_from_records(
    records: list[dict],
    train_percentage: int,
    protocol: str,
    validation_fraction: float = 0.1,
) -> tuple[list[int], list[int], list[int]]:
    """Create text-model split indices while preserving add-date/commit tie groups."""

    if protocol == "fixed-test":
        split = make_fixed_test_temporal_splits(records, train_percentages=[train_percentage])[0]
        return (
            list(range(split.train_start_index, split.train_end_index + 1)),
            list(range(split.validation_start_index, split.validation_end_index + 1)),
            list(range(split.test_start_index, split.test_end_index + 1)),
        )
    return make_text_protocol_indices(
        len(records),
        train_percentage=train_percentage,
        protocol=protocol,
        validation_fraction=validation_fraction,
    )


def tokenize_words(text: str) -> list[str]:
    """Tokenize SATD comments for TextCNN with a lightweight word tokenizer."""

    return re.findall(r"[a-z0-9_]+", str(text).lower())


def build_vocab(texts: Iterable[str], max_words: int) -> dict[str, int]:
    """Build a TextCNN vocabulary from training texts only."""

    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize_words(text))
    vocab = {"<PAD>": PAD_INDEX, "<UNK>": UNKNOWN_INDEX}
    for token, _count in counter.most_common(max(0, max_words - 2)):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_texts(texts: Iterable[str], vocab: dict[str, int], max_len: int) -> list[list[int]]:
    """Encode and pre-pad texts to a fixed length for TextCNN."""

    encoded_rows: list[list[int]] = []
    for text in texts:
        tokens = [vocab.get(token, UNKNOWN_INDEX) for token in tokenize_words(text)]
        tokens = tokens[-max_len:]
        padding = [PAD_INDEX] * max(0, max_len - len(tokens))
        encoded_rows.append(padding + tokens)
    return encoded_rows


def require_torch():
    """Import PyTorch lazily and report missing dependency clearly."""

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: torch") from exc
    return {"torch": torch, "nn": nn, "DataLoader": DataLoader, "Dataset": Dataset}


def require_transformers():
    """Import Transformers lazily and report missing dependency clearly."""

    try:
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("Missing dependency: transformers") from exc
    return {
        "AutoModel": AutoModel,
        "AutoTokenizer": AutoTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when available."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def metric_dict(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict:
    """Compute the shared classification metrics for text models."""

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else math.nan
    return {
        "auc": auc,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_textcnn_classes(torch_parts, config: TextCNNConfig):
    """Create TextCNN Dataset and model classes after PyTorch is available."""

    torch = torch_parts["torch"]
    nn = torch_parts["nn"]
    dataset_base = torch_parts["Dataset"]

    class TextDataset(dataset_base):
        """Serve encoded comments and binary labels to TextCNN."""

        def __init__(self, encoded_texts: list[list[int]], labels: list[int]):
            self.encoded_texts = encoded_texts
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return (
                torch.tensor(self.encoded_texts[index], dtype=torch.long),
                torch.tensor(self.labels[index], dtype=torch.float32),
            )

    class TextCNN(nn.Module):
        """Classify SATD comments with a multi-kernel convolutional text model."""

        def __init__(self, vocab_size: int):
            super().__init__()
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=config.embedding_dim,
                padding_idx=PAD_INDEX,
            )
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=config.embedding_dim,
                        out_channels=config.num_filters,
                        kernel_size=kernel_size,
                    )
                    for kernel_size in config.filter_sizes
                ]
            )
            self.dropout = nn.Dropout(config.dropout_rate)
            self.classifier = nn.Linear(config.num_filters * len(config.filter_sizes), 1)

        def forward(self, input_ids):
            embedded = self.embedding(input_ids).transpose(1, 2)
            pooled = [
                torch.relu(conv(embedded)).max(dim=2).values
                for conv in self.convs
            ]
            features = self.dropout(torch.cat(pooled, dim=1))
            return self.classifier(features).squeeze(1)

    return TextDataset, TextCNN


def run_textcnn_split(
    texts: list[str],
    labels: list[int],
    train_indices: list[int],
    validation_indices: list[int],
    test_indices: list[int],
    config: TextCNNConfig,
    device_name: str,
    seed: int,
) -> dict:
    """Train and evaluate TextCNN for one temporal split."""

    set_seed(seed)
    torch_parts = require_torch()
    torch = torch_parts["torch"]
    nn = torch_parts["nn"]
    DataLoader = torch_parts["DataLoader"]
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    TextDataset, TextCNN = build_textcnn_classes(torch_parts, config)

    vocab = build_vocab((texts[index] for index in train_indices), config.max_words)
    vocab_size = max(vocab.values()) + 1

    def encode_subset(indices: list[int]) -> tuple[list[list[int]], list[int]]:
        return (
            encode_texts((texts[index] for index in indices), vocab, config.max_len),
            [labels[index] for index in indices],
        )

    train_x, train_y = encode_subset(train_indices)
    valid_x, valid_y = encode_subset(validation_indices)
    test_x, test_y = encode_subset(test_indices)

    train_loader = DataLoader(
        TextDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TextDataset(valid_x, valid_y),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TextDataset(test_x, test_y),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = TextCNN(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_valid_loss = float("inf")
    patience_counter = 0

    for _epoch in range(config.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        valid_loss = evaluate_textcnn_loss(model, valid_loader, loss_fn, device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, y_score = predict_textcnn(model, test_loader, device)
    y_pred = [1 if score >= 0.5 else 0 for score in y_score]
    metrics = metric_dict(y_true, y_pred, y_score)
    metrics["vocab_size"] = vocab_size
    return metrics


def evaluate_textcnn_loss(model, dataloader, loss_fn, device) -> float:
    """Evaluate TextCNN validation loss."""

    total_loss = 0.0
    total_count = 0
    model.eval()
    with require_torch()["torch"].no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            total_loss += loss.item() * len(batch_y)
            total_count += len(batch_y)
    return total_loss / max(1, total_count)


def predict_textcnn(model, dataloader, device) -> tuple[list[int], list[float]]:
    """Predict TextCNN probabilities for a dataloader."""

    torch = require_torch()["torch"]
    y_true: list[int] = []
    y_score: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            scores = torch.sigmoid(model(batch_x)).detach().cpu().tolist()
            y_score.extend(float(score) for score in scores)
            y_true.extend(int(value) for value in batch_y.detach().cpu().tolist())
    return y_true, y_score


def build_bert_classes(torch_parts, transformer_parts, config: BertConfig):
    """Create BERT Dataset and classifier classes after dependencies are available."""

    torch = torch_parts["torch"]
    nn = torch_parts["nn"]
    dataset_base = torch_parts["Dataset"]
    AutoModel = transformer_parts["AutoModel"]

    class BertDataset(dataset_base):
        """Tokenize SATD comments for BERT fine-tuning."""

        def __init__(self, texts: list[str], labels: list[int], tokenizer):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            encoded = self.tokenizer(
                str(self.texts[index]),
                add_special_tokens=True,
                max_length=config.max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[index], dtype=torch.long),
            }

    class BertForSATD(nn.Module):
        """Fine-tune BERT with a projection layer and SATD repayment classifier."""

        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(config.model_name)
            hidden_size = self.bert.config.hidden_size
            self.down_proj = nn.Linear(hidden_size, config.embed_dim)
            self.dropout = nn.Dropout(config.dropout_rate)
            self.classifier = nn.Linear(config.embed_dim, 2)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            reduced = self.down_proj(cls_output)
            return self.classifier(self.dropout(reduced))

    return BertDataset, BertForSATD


def run_bert_split(
    texts: list[str],
    labels: list[int],
    train_indices: list[int],
    validation_indices: list[int],
    test_indices: list[int],
    config: BertConfig,
    device_name: str,
    seed: int,
) -> dict:
    """Fine-tune and evaluate BERT for one temporal split."""

    set_seed(seed)
    torch_parts = require_torch()
    transformer_parts = require_transformers()
    torch = torch_parts["torch"]
    nn = torch_parts["nn"]
    DataLoader = torch_parts["DataLoader"]
    AutoTokenizer = transformer_parts["AutoTokenizer"]
    get_scheduler = transformer_parts["get_linear_schedule_with_warmup"]
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    BertDataset, BertForSATD = build_bert_classes(torch_parts, transformer_parts, config)

    def subset(indices: list[int]) -> tuple[list[str], list[int]]:
        return [texts[index] for index in indices], [labels[index] for index in indices]

    train_texts, train_y = subset(train_indices)
    valid_texts, valid_y = subset(validation_indices)
    test_texts, test_y = subset(test_indices)

    train_loader = DataLoader(
        BertDataset(train_texts, train_y, tokenizer),
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        BertDataset(valid_texts, valid_y, tokenizer),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        BertDataset(test_texts, test_y, tokenizer),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = BertForSATD().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = get_scheduler(optimizer, 0, total_steps)
    loss_fn = nn.CrossEntropyLoss()
    best_state = None
    best_valid_loss = float("inf")
    patience_counter = 0

    for _epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_y = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        valid_loss = evaluate_bert_loss(model, valid_loader, loss_fn, device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, y_score = predict_bert(model, test_loader, device)
    y_pred = [1 if score >= 0.5 else 0 for score in y_score]
    return metric_dict(y_true, y_pred, y_score)


def evaluate_bert_loss(model, dataloader, loss_fn, device) -> float:
    """Evaluate BERT validation loss."""

    torch = require_torch()["torch"]
    total_loss = 0.0
    total_count = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * len(labels)
            total_count += len(labels)
    return total_loss / max(1, total_count)


def predict_bert(model, dataloader, device) -> tuple[list[int], list[float]]:
    """Predict BERT probabilities for a dataloader."""

    torch = require_torch()["torch"]
    y_true: list[int] = []
    y_score: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            y_score.extend(float(score) for score in scores)
            y_true.extend(int(value) for value in batch["label"].detach().cpu().tolist())
    return y_true, y_score


def format_metric(value: object) -> str:
    """Format metric values for CSV output."""

    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_metrics_csv(rows: list[dict], output_path: Path) -> None:
    """Write temporal text-model metrics."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protocol",
        "model",
        "train_percentage",
        "train_size",
        "validation_size",
        "test_size",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "max_train",
        "max_test",
        "auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "tn",
        "fp",
        "fn",
        "tp",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: format_metric(row.get(name, "")) for name in fieldnames})


def write_summary(rows: list[dict], output_path: Path, textcnn_config: TextCNNConfig, bert_config: BertConfig) -> None:
    """Write a Markdown summary for temporal text model trials."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = rows[0].get("protocol", DEFAULT_PROTOCOL) if rows else DEFAULT_PROTOCOL
    if protocol == "fixed-test":
        split_sentence = (
            "Splits use the chronological 8:1:1 protocol: the latest 10% of records "
            "is held out as a fixed test set, the preceding 10% is used for validation, "
            "and training windows expand from the earliest 10% to 80%. Boundaries "
            "are advanced when needed so one add-date/commit group is not split."
        )
    else:
        split_sentence = (
            "Splits follow the same global chronological order as the structured-feature "
            "temporal validation."
        )
    lines = [
        "# Temporal Text Model Summary",
        "",
        "## Setup",
        "",
        f"This experiment uses `f_comment` as the text input and `is_self_fixed` as the label. {split_sentence}",
        "",
        "TextCNN reuses the original hyperparameters: "
        f"MAX_WORDS={textcnn_config.max_words}, MAX_LEN={textcnn_config.max_len}, "
        f"EMB_DIM={textcnn_config.embedding_dim}, FILTER_SIZES={textcnn_config.filter_sizes}, "
        f"NUM_FILTERS={textcnn_config.num_filters}, DROPOUT={textcnn_config.dropout_rate}, "
        f"LR={textcnn_config.learning_rate}, BATCH={textcnn_config.batch_size}.",
        "",
        "BERT reuses the original hyperparameters: "
        f"MODEL={bert_config.model_name}, MAX_LEN={bert_config.max_len}, "
        f"EMBED_DIM={bert_config.embed_dim}, DROPOUT={bert_config.dropout_rate}, "
        f"LR={bert_config.learning_rate}, BATCH={bert_config.batch_size}.",
        "",
        "The TextCNN vocabulary is fit only on the training window for each temporal split, not on future test comments.",
        "",
        "## Results",
        "",
        "| Model | Train % | Train n | Validation n | Test n | AUC | F1 | MCC | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['train_percentage']}% | {row['train_size']:,} | "
            f"{row['validation_size']:,} | {row['test_size']:,} | "
            f"{row['auc']:.3f} | {row['f1']:.3f} | {row['mcc']:.3f} | {row.get('notes', '')} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_temporal_text_models(
    dataset_path: Path,
    metrics_path: Path,
    summary_path: Path,
    model_names: list[str],
    train_percentages: list[int],
    textcnn_config: TextCNNConfig,
    bert_config: BertConfig,
    protocol: str,
    validation_fraction: float,
    max_train: int | None,
    max_test: int | None,
    device: str,
    seed: int,
) -> list[dict]:
    """Run TextCNN and/or BERT on chronological temporal splits."""

    records = load_records(dataset_path)
    texts = [str(record.get("f_comment", "")) for record in records]
    labels = [int(record.get("is_self_fixed", 0)) for record in records]
    rows: list[dict] = []

    for train_percentage in train_percentages:
        if protocol == "remaining-test":
            full_train_indices, full_test_indices = make_split_indices(len(records), train_percentage)
            limited_train_indices = apply_sample_limit(full_train_indices, max_train)
            limited_test_indices = apply_sample_limit(full_test_indices, max_test)
            train_indices, validation_indices = train_validation_split(
                limited_train_indices, validation_fraction=validation_fraction
            )
        else:
            train_indices, validation_indices, full_test_indices = make_text_protocol_indices_from_records(
                records,
                train_percentage=train_percentage,
                protocol=protocol,
                validation_fraction=validation_fraction,
            )
            train_indices = apply_sample_limit(train_indices, max_train)
            validation_indices = apply_sample_limit(validation_indices, max_test)
            limited_test_indices = apply_sample_limit(full_test_indices, max_test)

        for model_name in model_names:
            try:
                if model_name == "textcnn":
                    metrics = run_textcnn_split(
                        texts,
                        labels,
                        train_indices,
                        validation_indices,
                        limited_test_indices,
                        config=textcnn_config,
                        device_name=device,
                        seed=seed,
                    )
                elif model_name == "bert":
                    metrics = run_bert_split(
                        texts,
                        labels,
                        train_indices,
                        validation_indices,
                        limited_test_indices,
                        config=bert_config,
                        device_name=device,
                        seed=seed,
                    )
                else:
                    raise ValueError(f"Unsupported model: {model_name}")
                notes = ""
            except Exception as exc:
                metrics = {
                    "auc": math.nan,
                    "accuracy": math.nan,
                    "precision": math.nan,
                    "recall": math.nan,
                    "f1": math.nan,
                    "mcc": math.nan,
                    "tn": "",
                    "fp": "",
                    "fn": "",
                    "tp": "",
                }
                notes = str(exc)

            row = {
                "protocol": protocol,
                "model": model_name,
                "train_percentage": train_percentage,
                "train_size": len(train_indices),
                "validation_size": len(validation_indices),
                "test_size": len(limited_test_indices),
                "train_start_date": records[train_indices[0]].get("add_date") if train_indices else "",
                "train_end_date": records[train_indices[-1]].get("add_date") if train_indices else "",
                "test_start_date": records[limited_test_indices[0]].get("add_date") if limited_test_indices else "",
                "test_end_date": records[limited_test_indices[-1]].get("add_date") if limited_test_indices else "",
                "max_train": max_train or "",
                "max_test": max_test or "",
                "notes": notes,
            }
            row.update(metrics)
            rows.append(row)
            if notes:
                print(f"{model_name} {train_percentage}% skipped: {notes}")
            else:
                print(
                    f"{model_name} {train_percentage}%: "
                    f"AUC={row['auc']:.3f}, F1={row['f1']:.3f}, MCC={row['mcc']:.3f}"
                )

    write_metrics_csv(rows, metrics_path)
    write_summary(rows, summary_path, textcnn_config, bert_config)
    return rows


def main() -> None:
    """Run temporal experiments for TextCNN and BERT on SATD comments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--models", default="textcnn,bert")
    parser.add_argument(
        "--protocol",
        choices=["remaining-test", "fixed-test"],
        default=DEFAULT_PROTOCOL,
        help="Use the legacy remaining-future test window or the 8:1:1 fixed-test protocol.",
    )
    parser.add_argument("--train-percentages", default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--textcnn-epochs", type=int, default=TEXTCNN_CONFIG.epochs)
    parser.add_argument("--textcnn-batch-size", type=int, default=TEXTCNN_CONFIG.batch_size)
    parser.add_argument("--bert-model", default=BERT_CONFIG.model_name)
    parser.add_argument("--bert-epochs", type=int, default=BERT_CONFIG.epochs)
    parser.add_argument("--bert-batch-size", type=int, default=BERT_CONFIG.batch_size)
    args = parser.parse_args()
    if args.train_percentages:
        train_percentages = parse_percentages(args.train_percentages)
    elif args.protocol == "fixed-test":
        train_percentages = list(DEFAULT_FIXED_TEST_TRAIN_PERCENTAGES)
    else:
        train_percentages = [80]

    textcnn_config = TextCNNConfig(
        epochs=args.textcnn_epochs,
        batch_size=args.textcnn_batch_size,
    )
    bert_config = BertConfig(
        model_name=args.bert_model,
        epochs=args.bert_epochs,
        batch_size=args.bert_batch_size,
    )
    run_temporal_text_models(
        dataset_path=args.dataset,
        metrics_path=args.metrics_csv,
        summary_path=args.summary,
        model_names=parse_models(args.models),
        train_percentages=train_percentages,
        textcnn_config=textcnn_config,
        bert_config=bert_config,
        protocol=args.protocol,
        validation_fraction=args.validation_fraction,
        max_train=args.max_train,
        max_test=args.max_test,
        device=args.device,
        seed=args.seed,
    )
    print(f"Wrote text metrics CSV: {args.metrics_csv}")
    print(f"Wrote text summary: {args.summary}")


if __name__ == "__main__":
    main()
