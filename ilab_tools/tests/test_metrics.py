from ilab_tools.metrics import ndcg_score


def test_ndcg_score():
    assert ndcg_score(['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'], 4) == 1
    assert ndcg_score(['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H'], 4) == 0
    assert ndcg_score(['A', 'B', 'C', 'D'], ['A', 'B', 'X', 'Y'], 2) == 1
    assert ndcg_score(['A', 'B', 'C', 'D'], ['E', 'F', 'C', 'D'], 2) == 0
