import math

def recall_at_k(ranklist, gold, k):
    s=set(gold)
    return 1.0 if any(r in s for r in ranklist[:k]) else 0.0

def mrr(ranklist, gold):
    s=set(gold)
    for i, r in enumerate(ranklist, 1):
        if r in s: return 1.0/i
    return 0.0

def ndcg_at_k(ranklist, gold, k):
    s=set(gold)
    dcg = 0.0
    for i, r in enumerate(ranklist[:k]):
        if r in s:
            dcg += 1.0 / math.log2(i+2)
    idcg = 1.0  # one relevant item assumed
    return dcg / idcg
