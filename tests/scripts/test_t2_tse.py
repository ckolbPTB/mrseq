"""Turbo-spin echo sequence for T2 mapping."""

import pytest
from mrseq.scripts.t2_tse import main as create_seq

EXPECTED_DUR = 512.000000  # defined 2025-10-08


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)


def test_seq_creation_error_on_short_te(system_defaults):
    """Test if error is raised on too short echo time."""
    with pytest.raises(ValueError):
        create_seq(system=system_defaults, te=1e-3, show_plots=False)


def test_seq_duration_vary_params_without_effect(system_defaults):
    """Test if sequence duration is as expected."""
    seq = create_seq(
        system=system_defaults,
        n_echoes=20,  # default 10
        fov_z=6e-3,  # default 8e-3
        show_plots=False,
        test_report=False,
        timing_check=False,
    )
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)
