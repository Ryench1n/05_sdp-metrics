"""Evaluation metrics for software defect prediction.

Хоёр бүлэг хэмжүүр:
  * decision-based — threshold-оор гаргасан хоёртын шийдвэрээс (confusion matrix)
  * ranking-based  — score-оор эрэмбэлсэн дарааллаас (threshold шаардахгүй)

PofB10, PofB20, PofB50 нь LOC-ийг effort-ийн төлөөлөл болгон ашигладаг хэмжүүрүүд
(кодын 10, 20, 50 хувийг шалгах төсөв).
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix,
)

DECISION_METRICS = ["Accuracy", "Precision", "Recall", "F1", "MCC", "G-mean"]
RANKING_METRICS = ["ROC-AUC", "PR-AUC", "PofB10", "PofB20", "PofB50", "IFA"]
POFB_BUDGETS = {"PofB10": 0.10, "PofB20": 0.20, "PofB50": 0.50}


def g_mean(y_true, y_pred):
    """Geometric mean of sensitivity (TPR) and specificity (TNR)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float(np.sqrt(tpr * tnr))


def pofb(y_true, score, loc, budget=0.20):
    """Proportion of defects found when inspecting `budget` share of total LOC.

    Модулиудыг score-оор буурахаар эрэмбэлж, хуримтлагдсан LOC нь
    budget * ΣLOC-ээс хэтрэхгүй модулиудыг шалгасан гэж үзнэ.
    Тэнцүү score-той модулиудын дараалал тогтвортой (mergesort).
    """
    y_true, score, loc = (np.asarray(a) for a in (y_true, score, loc))
    order = np.argsort(-score, kind="mergesort")
    within = np.cumsum(loc[order]) <= budget * loc.sum()
    total = y_true.sum()
    return float(y_true[order][within].sum() / total) if total else 0.0


def ifa(y_true, score):
    """Initial False Alarms: clean modules ranked above the first defective one."""
    y_true, score = np.asarray(y_true), np.asarray(score)
    ranked = y_true[np.argsort(-score, kind="mergesort")]
    hits = np.flatnonzero(ranked == 1)
    return int(hits[0]) if hits.size else int(ranked.size)


def evaluate(y_true, y_pred, score, loc):
    """Бүх хэмжүүрийг нэг дор тооцоолно.

    y_true : бодит шошго (0/1)
    y_pred : хоёртын шийдвэр (decision-based хэмжүүрт)
    score  : ranking-д ашиглах оноо (ranking-based хэмжүүрт)
    loc    : модуль бүрийн lines of code
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "G-mean": g_mean(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, score),
        "PR-AUC": average_precision_score(y_true, score),
        **{name: pofb(y_true, score, loc, b) for name, b in POFB_BUDGETS.items()},
        "IFA": ifa(y_true, score),
    }


def confusion(y_true, y_pred):
    """Return confusion matrix cells as a dict (TP, FP, FN, TN)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}
