from distill_eval.run import run


def test_run_synthetic() -> None:
    rep = run(["ignored"], random_state=0, synthetic=True)
    assert 0.0 <= rep.teacher_acc <= 1.0
    assert 0.0 <= rep.student_acc <= 1.0
    assert rep.teacher_f1 > 0.5
