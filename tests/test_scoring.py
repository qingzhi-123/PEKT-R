from pektr.scoring import recommend_from_state, score_one_candidate


def test_score_prefers_weak_knowledge_tag():
    state = [0.1, 0.9]
    weakness = [1.0, 0.0]
    a = {
        "problem_index": 1,
        "problem_id": "a",
        "title": "A",
        "tags": [0],
        "difficulty": 0.2,
        "error_cover": [1.0, 0.0],
        "tag_names": ["x"],
    }
    b = {
        "problem_index": 2,
        "problem_id": "b",
        "title": "B",
        "tags": [1],
        "difficulty": 0.8,
        "error_cover": [0.0, 0.0],
        "tag_names": ["y"],
    }
    assert score_one_candidate(state, weakness, a).score > score_one_candidate(state, weakness, b).score
    assert recommend_from_state(state, weakness, [b, a], top_n=1)[0].problem_id == "a"

