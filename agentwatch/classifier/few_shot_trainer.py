"""
Few-shot classifier training: the "weak system" that gets bootstrapped.

This is the core of what Raindrop calls "bespoke few-shot classifiers."
We take LLM-generated labels (expensive, slow) and distill them into a
small, fast model that can run on millions of traces cheaply.

Three approaches implemented:
1. SetFit — best few-shot performance, trains in seconds
2. Embedding + logistic regression — simplest, most interpretable
3. Prototype network — best for very few examples (5-10)

All produce a classifier that takes trace text → (label, confidence).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from agentwatch.config import get_settings


@dataclass
class TrainingResult:
    """Results from training a few-shot classifier."""

    issue_id: str
    approach: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_samples: int
    eval_samples: int
    training_time_ms: float
    inference_time_per_trace_ms: float
    model_size_mb: float
    device: str
    detailed_report: str = ""


@dataclass
class TrainedClassifier:
    """A trained classifier ready for inference."""

    issue_id: str
    approach: str
    model: object  # SetFitModel, LogisticRegression, or PrototypeClassifier
    encoder: SentenceTransformer | None = None
    metrics: TrainingResult | None = None

    def predict(self, texts: list[str]) -> list[tuple[bool, float]]:
        """Classify traces → list of (label, confidence)."""
        if self.approach == "setfit":
            preds = self.model.predict(texts)
            probs = self.model.predict_proba(texts)
            return [(bool(p), float(max(prob))) for p, prob in zip(preds, probs)]
        elif self.approach == "embedding_lr":
            embeddings = self.encoder.encode(texts, show_progress_bar=False)
            preds = self.model.predict(embeddings)
            probs = self.model.predict_proba(embeddings)
            return [(bool(p), float(max(prob))) for p, prob in zip(preds, probs)]
        elif self.approach == "prototype":
            return self.model.predict(texts)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")

    def predict_single(self, text: str) -> tuple[bool, float]:
        return self.predict([text])[0]


class PrototypeClassifier:
    """
    Prototype network: compute class centroids from few examples,
    classify by distance to nearest centroid. Works with as few as 3-5 examples.
    """

    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder
        self.pos_centroid: np.ndarray | None = None
        self.neg_centroid: np.ndarray | None = None

    def fit(self, texts: list[str], labels: list[bool]) -> None:
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        pos_embs = [e for e, l in zip(embeddings, labels) if l]
        neg_embs = [e for e, l in zip(embeddings, labels) if not l]
        self.pos_centroid = np.mean(pos_embs, axis=0)
        self.neg_centroid = np.mean(neg_embs, axis=0)

    def predict(self, texts: list[str]) -> list[tuple[bool, float]]:
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        results = []
        for emb in embeddings:
            d_pos = np.linalg.norm(emb - self.pos_centroid)
            d_neg = np.linalg.norm(emb - self.neg_centroid)
            # Softmax-style confidence
            exp_pos = np.exp(-d_pos)
            exp_neg = np.exp(-d_neg)
            conf = exp_pos / (exp_pos + exp_neg)
            results.append((bool(conf > 0.5), float(conf if conf > 0.5 else 1 - conf)))
        return results


def _split_data(texts: list[str], labels: list[bool], eval_ratio: float = 0.2):
    """Split into train/eval sets, stratified."""
    pos_idx = [i for i, l in enumerate(labels) if l]
    neg_idx = [i for i, l in enumerate(labels) if not l]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    n_pos_eval = max(1, int(len(pos_idx) * eval_ratio))
    n_neg_eval = max(1, int(len(neg_idx) * eval_ratio))

    eval_idx = set(pos_idx[:n_pos_eval] + neg_idx[:n_neg_eval])
    train_idx = [i for i in range(len(texts)) if i not in eval_idx]

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    eval_texts = [texts[i] for i in eval_idx]
    eval_labels = [labels[i] for i in eval_idx]

    return train_texts, train_labels, eval_texts, eval_labels


def _compute_metrics(
    y_true: list[bool],
    y_pred: list[bool],
    issue_id: str,
    approach: str,
    train_n: int,
    eval_n: int,
    train_time: float,
    inf_time: float,
    model_size: float,
    device: str,
) -> TrainingResult:
    report = classification_report(y_true, y_pred, target_names=["no_issue", "has_issue"])
    return TrainingResult(
        issue_id=issue_id,
        approach=approach,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        train_samples=train_n,
        eval_samples=eval_n,
        training_time_ms=train_time,
        inference_time_per_trace_ms=inf_time,
        model_size_mb=model_size,
        device=device,
        detailed_report=report,
    )


def train_setfit(
    texts: list[str],
    labels: list[bool],
    issue_id: str,
) -> TrainedClassifier:
    """
    Train a SetFit classifier — the best approach for few-shot text classification.
    SetFit trains a sentence-transformer with contrastive learning on few examples,
    then fits a classification head. Typically 8-64 examples needed.
    """
    settings = get_settings()
    train_texts, train_labels, eval_texts, eval_labels = _split_data(texts, labels)

    train_ds = Dataset.from_dict({"text": train_texts, "label": [int(l) for l in train_labels]})
    eval_ds = Dataset.from_dict({"text": eval_texts, "label": [int(l) for l in eval_labels]})

    model = SetFitModel.from_pretrained(
        settings.setfit_base_model,
        device=settings.device,
    )

    args = TrainingArguments(
        batch_size=min(16, len(train_texts)),
        num_epochs=2,
        num_iterations=20,
        output_dir=f"./models/{issue_id}_setfit",
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)

    t0 = time.time()
    trainer.train()
    train_time = (time.time() - t0) * 1000

    # Evaluate
    t1 = time.time()
    preds = model.predict(eval_texts)
    inf_time = (time.time() - t1) * 1000 / max(len(eval_texts), 1)

    eval_preds = [bool(p) for p in preds]

    # Estimate model size
    model_dir = f"./models/{issue_id}_setfit"
    model_size = 0
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            fp = os.path.join(model_dir, f)
            if os.path.isfile(fp):
                model_size += os.path.getsize(fp)
    model_size_mb = model_size / (1024 * 1024) if model_size > 0 else 22.0  # ~22MB for MiniLM

    metrics = _compute_metrics(
        eval_labels,
        eval_preds,
        issue_id,
        "setfit",
        len(train_texts),
        len(eval_texts),
        train_time,
        inf_time,
        model_size_mb,
        settings.device,
    )

    return TrainedClassifier(issue_id=issue_id, approach="setfit", model=model, metrics=metrics)


def train_embedding_lr(
    texts: list[str],
    labels: list[bool],
    issue_id: str,
) -> TrainedClassifier:
    """
    Embedding + logistic regression — simplest distillation approach.
    Encode all traces with sentence-transformer, fit sklearn LR.
    Fast, interpretable, works well with 30+ examples.
    """
    settings = get_settings()
    encoder = SentenceTransformer(settings.embedding_model, device=settings.device)

    train_texts, train_labels, eval_texts, eval_labels = _split_data(texts, labels)

    t0 = time.time()
    train_embs = encoder.encode(train_texts, show_progress_bar=False)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(train_embs, train_labels)
    train_time = (time.time() - t0) * 1000

    t1 = time.time()
    eval_embs = encoder.encode(eval_texts, show_progress_bar=False)
    preds = clf.predict(eval_embs)
    inf_time = (time.time() - t1) * 1000 / max(len(eval_texts), 1)

    eval_preds = [bool(p) for p in preds]

    metrics = _compute_metrics(
        eval_labels,
        eval_preds,
        issue_id,
        "embedding_lr",
        len(train_texts),
        len(eval_texts),
        train_time,
        inf_time,
        0.5,
        settings.device,  # LR model is tiny, encoder is shared
    )

    return TrainedClassifier(
        issue_id=issue_id,
        approach="embedding_lr",
        model=clf,
        encoder=encoder,
        metrics=metrics,
    )


def train_prototype(
    texts: list[str],
    labels: list[bool],
    issue_id: str,
) -> TrainedClassifier:
    """
    Prototype network — works with as few as 5 examples.
    Computes class centroids and classifies by distance.
    """
    settings = get_settings()
    encoder = SentenceTransformer(settings.embedding_model, device=settings.device)

    train_texts, train_labels, eval_texts, eval_labels = _split_data(texts, labels)

    proto = PrototypeClassifier(encoder)

    t0 = time.time()
    proto.fit(train_texts, train_labels)
    train_time = (time.time() - t0) * 1000

    t1 = time.time()
    results = proto.predict(eval_texts)
    inf_time = (time.time() - t1) * 1000 / max(len(eval_texts), 1)

    eval_preds = [r[0] for r in results]

    metrics = _compute_metrics(
        eval_labels,
        eval_preds,
        issue_id,
        "prototype",
        len(train_texts),
        len(eval_texts),
        train_time,
        inf_time,
        0.1,
        settings.device,
    )

    return TrainedClassifier(
        issue_id=issue_id,
        approach="prototype",
        model=proto,
        encoder=encoder,
        metrics=metrics,
    )


def train_all_approaches(
    texts: list[str],
    labels: list[bool],
    issue_id: str,
) -> dict[str, TrainedClassifier]:
    """Train all three approaches and return results for comparison."""
    results = {}
    for name, train_fn in [
        ("setfit", train_setfit),
        ("embedding_lr", train_embedding_lr),
        ("prototype", train_prototype),
    ]:
        try:
            results[name] = train_fn(texts, labels, issue_id)
        except Exception as e:
            print(f"Warning: {name} training failed: {e}")
    return results

