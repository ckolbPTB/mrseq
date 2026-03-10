"""Tests for Actual Flip Angle (AFI) GRE sequence for B1 mapping."""

import pytest
from mrseq.scripts.b1_afi_gre_dual_tr import main as create_seq

EXPECTED_DUR = 17.76  # defined 2026-03-04


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq, _ = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR, rel=1e-3)


def test_seq_creation_error_on_short_te(system_defaults):
    """Test if error is raised on too short echo time."""
    with pytest.raises(ValueError, match='TE must be larger'):
        create_seq(
            system=system_defaults,
            te=1e-3,  # Unrealistically short
            show_plots=False,
        )


def test_seq_creation_error_on_short_tr(system_defaults):
    """Test if error is raised on too short repetition time."""
    with pytest.raises(ValueError, match='TR must be larger than'):
        create_seq(
            system=system_defaults,
            tr1=2e-3,  # Too short for full sequence
            show_plots=False,
        )
