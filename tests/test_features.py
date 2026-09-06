from pektr.features import build_error_vector, temporal_decay_error_weakness, ERROR_INDEX


def test_compile_error_sets_syntax_label():
    vec = build_error_vector(
        judge_status="-2",
        info="{}",
        code="int main(){ return 0 }",
        language="C",
        tag_names=[],
    )
    assert vec[ERROR_INDEX["syntax_structure_error"]] == 1.0


def test_temporal_decay_error_weakness():
    weakness = temporal_decay_error_weakness([[1.0, 0.0], [0.0, 1.0]], rho=0.5)
    assert weakness == [0.5, 1.0]

