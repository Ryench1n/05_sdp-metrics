"""metrics.py-ийн зөв ажиллагааг шалгах unit test-үүд.

Ажиллуулах:  python -m pytest -q
"""
import numpy as np
import pytest

from metrics import evaluate, g_mean, ifa, pofb, DECISION_METRICS, RANKING_METRICS


# --- PofB -------------------------------------------------------------------

def test_pofb_hand_computed():
    # 5 модуль, нийт 1000 мөр, 20% төсөв = 200 мөр
    loc = np.array([100, 100, 400, 300, 100])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([0, 1, 1, 1, 0])
    # M0(100, clean) + M1(100, defective) = 200 -> төсөв дүүрэв; 1/3 defect
    assert pofb(y, score, loc, 0.20) == pytest.approx(1 / 3)


def test_pofb_large_module_eats_budget():
    loc = np.array([100, 100, 400, 300, 100])
    y = np.array([0, 1, 1, 1, 0])
    score = np.array([0.1, 0.9, 0.8, 0.7, 0.2])
    # M1(100) дараа нь M2(400) -> 500 > 200 тул зөвхөн M1 багтана
    assert pofb(y, score, loc, 0.20) == pytest.approx(1 / 3)


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
    loc = rng.integers(20, 2000, n).astype(float)
    return y, prob, loc


def test_threshold_change_leaves_ranking_metrics_unchanged(toy):
    y, prob, loc = toy
    a = evaluate(y, (prob >= 0.5).astype(int), prob, loc)
    b = evaluate(y, (prob >= 0.3).astype(int), prob, loc)
    for m in RANKING_METRICS:
        assert a[m] == pytest.approx(b[m]), m
    assert any(a[m] != pytest.approx(b[m]) for m in DECISION_METRICS)


def test_ranking_change_leaves_decision_metrics_unchanged(toy):
    y, prob, loc = toy
    pred = (prob >= 0.5).astype(int)
    a = evaluate(y, pred, prob, loc)
    b = evaluate(y, pred, prob / loc, loc)
    for m in DECISION_METRICS:
        assert a[m] == pytest.approx(b[m]), m
    assert any(a[m] != pytest.approx(b[m]) for m in RANKING_METRICS)


# --- Data loader ------------------------------------------------------------

def test_nodes_loader_joins_loc_and_excludes_targets():
    from data import ADVISOR_DIR, load_nodes
    if not (ADVISOR_DIR / "ant-1.7_nodes.csv").exists():
        pytest.skip("advisor data not present")
    X, y, loc, features, df = load_nodes("ant-1.7")
    assert X.shape == (744, len(features))
    assert y.sum() == 165
    assert not np.isnan(loc).any()
    assert not {"bug", "label", "loc"} & set(features)
    assert (y == (df["bug"] > 0)).all()
