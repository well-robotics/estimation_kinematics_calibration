"""Optional integration paths stay portable."""

from estimation_calibration_cuda.data_paths import (
    leg_bical_data_root,
    leg_bical_golden_path,
    optional_env_path,
)


def test_optional_path_is_none_when_unset_or_blank(monkeypatch):
    monkeypatch.delenv("LEG_BICAL_TEST_PATH", raising=False)
    assert optional_env_path("LEG_BICAL_TEST_PATH") is None
    monkeypatch.setenv("LEG_BICAL_TEST_PATH", "  ")
    assert optional_env_path("LEG_BICAL_TEST_PATH") is None


def test_optional_paths_expand_user(monkeypatch):
    monkeypatch.setenv("LEG_BICAL_DATA_ROOT", "~/datasets_v0")
    monkeypatch.setenv("LEG_BICAL_GOLDEN_PATH", "~/golden.npz")
    assert leg_bical_data_root().is_absolute()
    assert leg_bical_data_root().name == "datasets_v0"
    assert leg_bical_golden_path().is_absolute()
    assert leg_bical_golden_path().name == "golden.npz"
