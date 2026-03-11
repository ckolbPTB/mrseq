"""Tests for the diffusion preparation block."""

import pytest
from mrseq.preparations.diffusion_prep import DiffusionPrep
from mrseq.preparations.diffusion_prep import calculate_b_value
from mrseq.preparations.diffusion_prep import calculate_diffusion_gradient_duration


@pytest.mark.parametrize('g_amplitude', (4e5, 8e5, 1e6, 3e6))
@pytest.mark.parametrize('g_duration', (5e-6, 10e-6, 20e-6, 40e-6))
@pytest.mark.parametrize('g_delta_time', (20e-6, 40e-6, 80e-6, 100e-6))
@pytest.mark.parametrize('g_rise_time', (0.1e-6, 0.2e-6, 0.5e-6, 1e-6))
def test_duration_calculation(g_amplitude, g_duration, g_rise_time, g_delta_time):
    """Test the gradient duration calculation."""
    if g_duration < g_delta_time:
        b_value = calculate_b_value(g_amplitude, g_duration, g_rise_time, g_delta_time)
        g_duration_calc = calculate_diffusion_gradient_duration(b_value, g_amplitude, g_delta_time)
        assert g_duration == pytest.approx(g_duration_calc, rel=1e-3)


@pytest.mark.parametrize('g_amplitude', (4e5, 8e5, 1e6))
@pytest.mark.parametrize('g_delta_time', (40e-6, 80e-6))
def test_block_duration(system_defaults, g_amplitude, g_delta_time):
    """Test the block duration calculation."""
    diff_prep = DiffusionPrep(system_defaults, g_amplitude=g_amplitude, g_delta_time=g_delta_time)
    seq, block_duration, _ = diff_prep.add_diffusion_prep(seq=None, b_balue=200e-3)
    seq_b0, block_duration_b0, _ = diff_prep.add_diffusion_prep(seq=None, b_balue=0)
    assert block_duration == block_duration_b0
    assert sum(seq.block_durations.values()) == sum(seq_b0.block_durations.values())
