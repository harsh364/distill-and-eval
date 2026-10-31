from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


@dataclass
class Report:
    teacher_acc: float
    teacher_f1: float
    student_acc: float
    student_f1: float


def _synthetic_corpus(n_per_class: int = 50, random_state: int = 0) -> tuple[list[str], np.ndarray]:
    """Offline-friendly stand-in for tutorial/CI (two topic templates)."""
    rng = np.random.default_rng(random_state)
    templates = [
        "patient hospital medical clinical trial drug therapy health",
        "orbit satellite launch space mission astronomy planet",
    ]
    texts: list[str] = []
    labels: list[int] = []
    for label, stem in enumerate(templates):
        for i in range(n_per_class):
            texts.append(f"{stem} sample document number {i} noise{rng.integers(99999)}")
            labels.append(label)
    return texts, np.array(labels)


def run(
    categories: list[str],
    random_state: int = 42,
    *,
    synthetic: bool = False,
) -> Report:
    if synthetic:
        texts, y = _synthetic_corpus(random_state=random_state)
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=y,
        )
    else:
        data = fetch_20newsgroups(
            subset="all",
            categories=categories,
            remove=("headers", "footers", "quotes"),
        )
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.2,
            random_state=random_state,
            stratify=data.target,
        )

    teacher_vec = TfidfVectorizer(max_features=5000, stop_words="english")
    x_train_t = teacher_vec.fit_transform(x_train)
    x_test_t = teacher_vec.transform(x_test)

    teacher = LogisticRegression(max_iter=2000)
    teacher.fit(x_train_t, y_train)
    teacher_probs = teacher.predict_proba(x_test_t)
    teacher_pred = teacher_probs.argmax(axis=1)

    svd = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            ("svd", TruncatedSVD(n_components=64, random_state=random_state)),
            ("norm", Normalizer(copy=False)),
        ]
    )
    x_train_s = svd.fit_transform(x_train)
    x_test_s = svd.transform(x_test)

    teacher_train_probs = teacher.predict_proba(x_train_t)
    student = Ridge(alpha=1.0)
    student.fit(x_train_s, teacher_train_probs)
    student_probs = student.predict(x_test_s)
    student_probs = np.clip(student_probs, 1e-6, None)
    student_probs = student_probs / student_probs.sum(axis=1, keepdims=True)
    student_pred = student_probs.argmax(axis=1)

    return Report(
        teacher_acc=float(accuracy_score(y_test, teacher_pred)),
        teacher_f1=float(f1_score(y_test, teacher_pred, average="macro")),
        student_acc=float(accuracy_score(y_test, student_pred)),
        student_f1=float(f1_score(y_test, student_pred, average="macro")),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--categories",
        nargs="+",
        default=["sci.med", "sci.space", "rec.sport.baseball"],
    )
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use built-in two-topic corpus (offline, no download)",
    )
    args = p.parse_args()
    rep = run(args.categories, synthetic=args.synthetic)
    payload = {
        "teacher": {"accuracy": rep.teacher_acc, "macro_f1": rep.teacher_f1},
        "student": {"accuracy": rep.student_acc, "macro_f1": rep.student_f1},
        "note": "Student uses 64-dim SVD features matching teacher probabilities",
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    print(f"teacher accuracy={rep.teacher_acc:.3f} macro_f1={rep.teacher_f1:.3f}")
    print(
        f"student accuracy={rep.student_acc:.3f} macro_f1={rep.student_f1:.3f} "
        "(64-dim student vs 5000-dim teacher features)"
    )


if __name__ == "__main__":
    main()
