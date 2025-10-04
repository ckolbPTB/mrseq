"""Tests for 3D FLASH sequence with golden radial phase encoding."""

import pytest
from mrseq.scripts.grpe_flash_dixon import main as create_seq

EXPECTED_DUR = 7.96752  # defined 2025-09-22


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)


def test_seq_creation_error_on_short_tr(system_defaults):
    """Test if error is raised on too short repetition time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, tr=5e-3, show_plots=False)


def test_seq_duration_vary_params_without_effect(system_defaults):
    """Test if sequence duration is as expected."""
    seq = create_seq(
        system=system_defaults,
        n_rpe_points_per_shot=4,  # default 8
        show_plots=False,
        test_report=False,
        timing_check=False,
    )
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)


def test_fat_sat(system_defaults):
    """Test that fat sat leads to longer sequence."""
    seq = create_seq(system=system_defaults, show_plots=False, fat_saturation=True)
    duration = seq.duration()[0]
    assert duration > EXPECTED_DUR
