"""metrics.py-ийн зөв ажиллагааг шалгах unit test-үүд.

Ажиллуулах:  python -m pytest -q
"""
import numpy as np
import pytest

from metrics import evaluate, g_mean, ifa, pofb, DECISION_METRICS, RANKING_METRICS


# --- PofB -------------------------------------------------------------------

def test_pofb_hand_computed():
    # 5 модуль, нийт 1000 мөр, 20% төсөв = 200 мөр
    effort = np.array([100, 100, 400, 300, 100])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([0, 1, 1, 1, 0])
    # M0(100, clean) + M1(100, defective) = 200 -> төсөв дүүрэв; 1/3 defect
    assert pofb(y, score, effort, 0.20) == pytest.approx(1 / 3)


def test_pofb_large_module_eats_budget():
    effort = np.array([100, 100, 400, 300, 100])
    y = np.array([0, 1, 1, 1, 0])
    score = np.array([0.1, 0.9, 0.8, 0.7, 0.2])
    # M1(100) дараа нь M2(400) -> 500 > 200 тул зөвхөн M1 багтана
    assert pofb(y, score, effort, 0.20) == pytest.approx(1 / 3)


def test_pofb_full_budget_finds_everything():
    y = np.array([1, 0, 1, 0])
    assert pofb(y, np.array([.4, .3, .2, .1]), np.array([10, 10, 10, 10]), 1.0) == 1.0


# --- IFA --------------------------------------------------------------------

def test_ifa_counts_false_alarms_before_first_defect():
    y = np.array([0, 1, 1, 1, 0])
    assert ifa(y, np.array([0.9, 0.8, 0.7, 0.6, 0.5])) == 1
    assert ifa(y, np.array([0.1, 0.9, 0.8, 0.7, 0.2])) == 0


def test_ifa_no_defects_returns_length():
    assert ifa(np.array([0, 0, 0]), np.array([.3, .2, .1])) == 3


# --- G-mean -----------------------------------------------------------------

def test_g_mean_perfect_and_zero():
    y = np.array([0, 1, 0, 1])
    assert g_mean(y, y) == pytest.approx(1.0)
    assert g_mean(y, np.zeros(4, int)) == 0.0   # sensitivity = 0


# --- Invariance -------------------------------------------------------------
# Судалгааны гол шинж чанар: decision-based ба ranking-based хэмжүүрүүд
# бие биеийнхээ өөрчлөлтийг огт мэдэрдэггүй.

@pytest.fixture
def toy():
    rng = np.random.default_rng(1)
    n = 200
    y = (rng.random(n) < 0.25).astype(int)
    prob = np.clip(0.3 * y + rng.random(n) * 0.7, 0, 1)
    effort = rng.integers(20, 2000, n).astype(float)
    return y, prob, effort


def test_threshold_change_leaves_ranking_metrics_unchanged(toy):
    y, prob, effort = toy
    a = evaluate(y, (prob >= 0.5).astype(int), prob, effort)
    b = evaluate(y, (prob >= 0.3).astype(int), prob, effort)
    for m in RANKING_METRICS:
        assert a[m] == pytest.approx(b[m]), m
    assert any(a[m] != pytest.approx(b[m]) for m in DECISION_METRICS)


def test_ranking_change_leaves_decision_metrics_unchanged(toy):
    y, prob, effort = toy
    pred = (prob >= 0.5).astype(int)
    a = evaluate(y, pred, prob, effort)
    b = evaluate(y, pred, prob / effort, effort)
    for m in DECISION_METRICS:
        assert a[m] == pytest.approx(b[m]), m
    assert any(a[m] != pytest.approx(b[m]) for m in RANKING_METRICS)


# --- Data loader ------------------------------------------------------------

def test_loader_promise_datasets():
    from data import DATA_DIR, DATASETS, load
    if not all((DATA_DIR / f"{d}.csv").exists() for d in DATASETS):
        pytest.skip("PROMISE data not present")
    X, y, effort, features, df = load("ant-1.7")
    assert X.shape == (745, 20)
    assert y.sum() == 166
    assert "loc" in features and "bug" not in features
    assert (effort == df["loc"]).all()


# --- PofB budgets -----------------------------------------------------------

def test_pofb_budgets_are_monotonic(toy):
    y, prob, effort = toy
    r = evaluate(y, (prob >= 0.5).astype(int), prob, effort)
    assert r["PofB10"] <= r["PofB20"] <= r["PofB50"]


# --- Statistics -------------------------------------------------------------

def test_holm_matches_hand_computation():
    from stats import holm
    # sorted p: 0.01, 0.02, 0.04 -> 0.03, 0.04, 0.04
    assert np.allclose(holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])


def test_cliffs_delta_extremes_and_symmetry():
    from stats import cliffs_delta
    assert cliffs_delta([3, 4, 5], [0, 1, 2]) == 1.0
    assert cliffs_delta([0, 1, 2], [3, 4, 5]) == -1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0


def test_wilcoxon_identical_samples_not_significant():
    from stats import wilcoxon_paired
    stat, p = wilcoxon_paired([0.5] * 10, [0.5] * 10)
    assert p == 1.0


def test_percentile_ci_contains_center():
    from stats import percentile_ci
    lo, hi = percentile_ci(np.arange(1001))
    assert lo == pytest.approx(25) and hi == pytest.approx(975)
