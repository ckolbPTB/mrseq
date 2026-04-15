"""Tests for SASHA with FLASH readout for T1 mapping."""

import numpy as np
import pytest
from mrseq.scripts.t1_sasha_flash import main as create_seq
from mrseq.utils.system_defaults import sys_a
from mrseq.utils.system_defaults import sys_b

EXPECTED_DUR = 7.00569  # defined 2026-04-14


def test_default_seq_duration(system_defaults):
    """Test if default values result in expected sequence duration."""
    seq, _ = create_seq(system=system_defaults, show_plots=False)
    duration = seq.duration()[0]
    assert duration == pytest.approx(EXPECTED_DUR)


@pytest.mark.parametrize('system', [sys_a, sys_b])
def test_seq_duration(system):
    """Test system dependance of sequence."""
    seq, _ = create_seq(system=system, show_plots=False)
    duration = seq.duration()[0]
    assert np.abs(duration - EXPECTED_DUR) / EXPECTED_DUR < 0.05
