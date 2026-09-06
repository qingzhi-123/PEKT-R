from pektr.metrics import hit_rate_at_k, ndcg_at_k


def test_hit_rate_and_ndcg():
    recs = {"u1": ["a", "b", "c"], "u2": ["d", "e"]}
    truth = {"u1": {"c"}, "u2": {"x"}}
    assert hit_rate_at_k(recs, truth, k=3) == 0.5
    assert 0.0 < ndcg_at_k(recs, truth, k=3) < 1.0

