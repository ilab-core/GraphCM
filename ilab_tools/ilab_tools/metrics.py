import math


def ndcg_score(true_labels, predicted_labels, k):

    def dcg(items, relevant_items, k):
        return sum((1.0 / math.log2(idx + 2) if items[idx] in relevant_items else 0) for idx in range(min(k, len(items))))

    ideal_items = true_labels[:k]
    ideal_dcg = dcg(ideal_items, true_labels, k)
    actual_dcg = dcg(predicted_labels, true_labels, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0


def count_different(true_labels, predicted_labels, tolerance=1e-6):
    return sum(abs(a - b) > tolerance for a, b in zip(true_labels, predicted_labels))
