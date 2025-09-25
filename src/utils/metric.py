# -*- coding:utf-8 -*-
# @FileName : metric.py
# @Time : 2025/2/9 14:52
# @Author : fiv

import os
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from Levenshtein import distance as levenshtein_distance
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
from typing import List, Dict
from tqdm import tqdm

# !pip install nltk python-Levenshtein

# import nltk

# nltk.download("popular")
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("punkt_tab")


def compute_edit_distance(pred: str, ref: str) -> int:
    """计算编辑距离（Levenshtein Distance）"""
    return levenshtein_distance(pred, ref)


def compute_normalized_edit_distance(pred: str, ref: str) -> float:
    """计算归一化的编辑距离（0-1范围）"""
    edit_dist = levenshtein_distance(pred, ref)
    max_len = max(len(pred), len(ref))
    return edit_dist / max_len if max_len > 0 else 0.0


def compute_f1_precision_recall(
    pred_tokens: List[str], ref_tokens: List[str]
) -> Dict[str, float]:
    """计算词级别的F1、Precision、Recall"""
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)

    tp = len(pred_set & ref_set)  # 正确匹配的词数
    fp = len(pred_set - ref_set)  # 预测多余词数
    fn = len(ref_set - pred_set)  # 未预测到的词数

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"f1": f1, "precision": precision, "recall": recall}


def compute_bleu(pred: str, ref: str) -> float:
    """计算句子级BLEU分数"""
    return sentence_bleu([ref], pred)


def compute_meteor(pred: str, ref: str) -> float:
    """计算METEOR分数（需要分词）"""
    pred_tokens = word_tokenize(pred)
    ref_tokens = word_tokenize(ref)
    return meteor_score.single_meteor_score(ref_tokens, pred_tokens)


def calculate_all_metrics(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """批量计算所有指标"""
    metrics = {
        "edit_distance": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "bleu": [],
        "meteor": [],
    }
    iteration = list(zip(predictions, references))

    for pred, ref in tqdm(
        iteration,
        desc=f"Calculating Metrics For {os.environ.get('RUNNAME', 'default')}",
    ):
        # Edit Distance
        metrics["edit_distance"].append(compute_normalized_edit_distance(pred, ref))

        # F1/Precision/Recall（基于词级）
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = word_tokenize(ref.lower())
        scores = compute_f1_precision_recall(pred_tokens, ref_tokens)
        metrics["f1"].append(scores["f1"])
        metrics["precision"].append(scores["precision"])
        metrics["recall"].append(scores["recall"])

        # BLEU
        metrics["bleu"].append(compute_bleu(pred, ref))

        # METEOR
        metrics["meteor"].append(compute_meteor(pred, ref))

    # 计算平均值
    return {
        "edit_distance↓": np.mean(metrics["edit_distance"]),
        "f1↑": np.mean(metrics["f1"]),
        "precision↑": np.mean(metrics["precision"]),
        "recall↑": np.mean(metrics["recall"]),
        "bleu↑": np.mean(metrics["bleu"]),
        "meteor↑": np.mean(metrics["meteor"]),
    }


if __name__ == "__main__":
    # 示例数据
    predictions = [
        "\\text{Secure mobile}\n\\text{edge caching}\\text{cent�}\\text{cent}\\text{edge}\\text{caching}\\text{cent}\\text{edge}\\text{caching}\\text{cent}\\text{edge}\\text{caching}\\text{edge}\\text{caching}\\text{edge}\\text{caching}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\text{edge}\\"
    ]
    references = [
        "\\begin{tabular}[c]{@{}l@{}} Secure mobile \\\\ edge  caching \\end{tabular}"
    ]
    # 计算指标
    results = calculate_all_metrics(predictions, references)
    print(results)


# class StructMetric:
#     pass


# class ContentMetric:
#     def edit_distance(self, str1, str2):
#         """
#         Formula: Minimum number of operations (insert, delete, replace) to convert one string to another.
#         Result: the smaller, the better
#         """
#         m, n = len(str1), len(str2)
#         dp = np.zeros((m + 1, n + 1), dtype=int)

#         for i in range(m + 1):
#             for j in range(n + 1):
#                 if i == 0:
#                     dp[i][j] = j
#                 elif j == 0:
#                     dp[i][j] = i
#                 elif str1[i - 1] == str2[j - 1]:
#                     dp[i][j] = dp[i - 1][j - 1]
#                 else:
#                     dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

#         return dp[m][n]

#     def jaccard_similarity(self, set1, set2):
#         """
#         Formula: Jaccard(A, B) = |A ∩ B| / |A ∪ B|
#         Result: the larger, the better
#         """
#         intersection = len(set1 & set2)
#         union = len(set1 | set2)
#         return intersection / union if union != 0 else 0

#     def cosine_similarity(self, vec1, vec2):
#         """
#         Formula: Cosine(A, B) = (A · B) / (||A|| * ||B||)
#         Result: the larger, the better
#         """
#         dot_product = np.dot(vec1, vec2)
#         norm1 = np.linalg.norm(vec1)
#         norm2 = np.linalg.norm(vec2)
#         return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

#     def bleu_score(self, reference, candidate):
#         """
#         Formula: BLEU = BP * exp(sum(w_n * log(p_n)))
#         Result: the larger, the better
#         """
#         return sentence_bleu([reference], candidate)
