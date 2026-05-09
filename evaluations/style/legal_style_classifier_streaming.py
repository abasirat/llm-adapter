#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Optional

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def stream_texts(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    text_field: str,
    max_samples: int,
    min_chars: int = 100,
    max_chars: Optional[int] = None,
) -> List[str]:
    if subset:
        ds = load_dataset(dataset_name, subset, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=True)

    texts = []

    for ex in ds:
        if text_field not in ex:
            continue

        text = ex[text_field]

        if text is None:
            continue

        text = str(text).strip()

        if len(text) < min_chars:
            continue

        if max_chars is not None:
            text = text[:max_chars]

        texts.append(text)

        if len(texts) >= max_samples:
            break

    return texts


def train_classifier(args):
    print("Streaming legal samples...")
    legal_texts = stream_texts(
        dataset_name=args.legal_dataset,
        subset=args.legal_subset,
        split=args.legal_split,
        text_field=args.legal_text_field,
        max_samples=args.max_samples,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    print("Streaming general samples...")
    general_texts = stream_texts(
        dataset_name=args.general_dataset,
        subset=args.general_subset,
        split=args.general_split,
        text_field=args.general_text_field,
        max_samples=args.max_samples,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    n = min(len(legal_texts), len(general_texts))

    if n == 0:
        raise ValueError("No usable samples were loaded. Check dataset names, splits, and text fields.")

    legal_texts = legal_texts[:n]
    general_texts = general_texts[:n]

    print(f"Loaded {n} legal samples and {n} general samples.")

    texts = legal_texts + general_texts
    labels = [1] * n + [0] * n

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    clf = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, args.max_ngram),
                    min_df=args.min_df,
                    max_df=args.max_df,
                    max_features=args.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=args.max_iter,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nValidation accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["general", "legal"]))

    os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)
    joblib.dump(clf, args.output_model)

    print(f"Saved classifier to: {args.output_model}")
    print_top_features(clf, args.top_k)


def print_top_features(clf, top_k: int):
    vectorizer = clf.named_steps["tfidf"]
    logreg = clf.named_steps["logreg"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = logreg.coef_[0]

    top_legal = np.argsort(coefs)[-top_k:][::-1]
    top_general = np.argsort(coefs)[:top_k]

    print("\nTop legal-associated features:")
    for idx in top_legal:
        print(f"{feature_names[idx]:<35} {coefs[idx]:.4f}")

    print("\nTop general-associated features:")
    for idx in top_general:
        print(f"{feature_names[idx]:<35} {coefs[idx]:.4f}")


def load_jsonl_texts(path: str, text_field: str):
    texts = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get(text_field, "")
            text = str(text).strip()
            if text:
                texts.append(text)

    return texts


def score_outputs(args):
    clf = joblib.load(args.model_file)
    texts = load_jsonl_texts(args.input_file, args.text_field)

    legal_probs = clf.predict_proba(texts)[:, 1] # Probability of legal class
    general_probs = clf.predict_proba(texts)[:, 0]
    preds = clf.predict(texts)

    results = []

    for text, general_prob, legal_prob, pred in zip(texts, general_probs, legal_probs, preds):
        results.append(
            {
                "text": text,
                "legal_style_score": float(legal_prob),
                "general_style_score": float(general_prob),
                "predicted_label": "legal" if pred == 1 else "general",
            }
        )

    summary = {
        "input_file": args.input_file,
        "num_examples": len(texts),
        "mean_legal_style_score": float(np.mean(legal_probs)),
        "median_legal_style_score": float(np.median(legal_probs)),
        "mean_general_style_score": float(np.mean(general_probs)),
        "median_general_style_score": float(np.median(general_probs)),
        "predicted_legal_rate": float(np.mean(preds)),
        "results": results,
    }

    print(f"\nExamples: {summary['num_examples']}")
    print(f"Mean LegalStyleScore:   {summary['mean_legal_style_score']:.4f}")
    print(f"Median LegalStyleScore: {summary['median_legal_style_score']:.4f}")
    print(f"Mean GeneralStyleScore: {summary['mean_general_style_score']:.4f}")
    print(f"Median GeneralStyleScore: {summary['median_general_style_score']:.4f}")
    print(f"Predicted legal rate:   {summary['predicted_legal_rate']:.4f}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved scores to: {args.output_file}")


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")

    train.add_argument("--legal_dataset", type=str, default="pile-of-law/pile-of-law")
    train.add_argument("--legal_subset", type=str, default=None)
    train.add_argument("--legal_split", type=str, default="train")
    train.add_argument("--legal_text_field", type=str, default="text")

    train.add_argument("--general_dataset", type=str, default="HuggingFaceFW/fineweb")
    train.add_argument("--general_subset", type=str, default="sample-10BT")
    train.add_argument("--general_split", type=str, default="train")
    train.add_argument("--general_text_field", type=str, default="text")

    train.add_argument("--max_samples", type=int, default=10000)
    train.add_argument("--min_chars", type=int, default=100)
    train.add_argument("--max_chars", type=int, default=2000)

    train.add_argument("--output_model", type=str, default="legal_style_clf.joblib")

    train.add_argument("--test_size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)

    train.add_argument("--max_ngram", type=int, default=2)
    train.add_argument("--min_df", type=int, default=3)
    train.add_argument("--max_df", type=float, default=0.95)
    train.add_argument("--max_features", type=int, default=100000)
    train.add_argument("--max_iter", type=int, default=1000)
    train.add_argument("--top_k", type=int, default=30)

    score = subparsers.add_parser("score")
    score.add_argument("--model_file", type=str, required=True)
    score.add_argument("--input_file", type=str, required=True)
    score.add_argument("--output_file", type=str, default="legal_style_scores.json")
    score.add_argument("--text_field", type=str, default="text")

    args = parser.parse_args()

    if args.command == "train":
        train_classifier(args)
    elif args.command == "score":
        score_outputs(args)


if __name__ == "__main__":
    main()