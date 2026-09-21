"""Tests for GMTF estimation and correction."""

import numpy as np
import pypulseq as pp
import pytest
from mrseq.utils.Gmtf import Gmtf
from mrseq.utils.Gmtf import build_input_triangles
from mrseq.utils.Gmtf import calc_kspace_from_grad_waveforms
from mrseq.utils.Gmtf import convert_waveforms_to_ppoly
from mrseq.utils.Gmtf import phase_to_gradient
from mrseq.utils.Gmtf import unwrap_phase_difference


def create_test_sequence() -> tuple[pp.Sequence, np.ndarray]:
    """
    Build a simple test sequence: RF excitation, a negative pre-winder followed by a readout gradient.

    Returns
    -------
    seq
        The assembled Pulseq sequence.
    trajectory
        Ground-truth k-space trajectory (1/m).
    dwell_time
        ADC dwell time
    """
    system = pp.Opts()
    seq = pp.Sequence(system=system)
    rf = pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, system=system)
    gx_pre = pp.make_trapezoid(channel='x', area=-300, system=system)
    gx = pp.make_trapezoid(channel='x', area=-2 * gx_pre.area, system=system)
    n_samples = 100
    adc = pp.make_adc(num_samples=n_samples, duration=gx.flat_time, delay=gx.rise_time, system=system)

    seq.add_block(rf)
    seq.add_block(pp.make_trapezoid(channel='x', area=0, duration=0.01, system=system))
    seq.add_block(gx_pre)
    seq.add_block(gx, adc)
    k_traj_adc = seq.calculate_kspace()[0]
    return seq, k_traj_adc, adc.dwell


def test_unwrap_phase_difference_removes_2pi_jumps():
    """Unwrap linearly increasing phase."""
    n = 200
    true_phase = np.linspace(0, 20 * np.pi, n)
    signal0 = np.exp(1j * true_phase)
    signal1 = np.zeros(n)
    data = np.stack([signal0, signal1])[None, :, None, None, :]
    result = unwrap_phase_difference(data)
    # Unwrapped result should have no jump > pi between consecutive samples
    assert np.all(np.abs(np.diff(result.squeeze())) < np.pi + 1e-6)
    # And should approximately recover the linear ramp (up to a constant offset)
    np.testing.assert_allclose(np.diff(result.squeeze()), np.diff(true_phase), atol=1e-3)


def test_unwrap_phase_difference_known_constant_offset():
    """Recover constant offset between two phase signlas."""
    n = 50
    offset = 1.3
    signal0 = np.exp(1j * offset) * np.ones(n)
    signal1 = np.ones(n)
    data = np.stack([signal0, signal1])[None, :, None, None, :]
    result = unwrap_phase_difference(data)
    np.testing.assert_allclose(result, offset, atol=1e-10)


def test_phase_to_gradient_known_linear_ramp():
    """A known linear phase ramp should produce a known constant gradient."""
    n = 100
    slope = 0.05  # rad per sample
    phase_mean = np.tile(slope * np.arange(n), (3, 5, 1))
    phase_std = np.zeros((3, 5, n))
    slice_pos, gamma, dwell_time = 0.01, 2.675e8, 1e-6
    grad_mean, _ = phase_to_gradient(phase_mean, phase_std, slice_pos, gamma, dwell_time)
    expected = slope / (slice_pos * gamma * dwell_time)
    np.testing.assert_allclose(grad_mean, expected, rtol=1e-10)


def test_build_input_triangles_peak_amplitude():
    """Peak of each triangle should equal enumerate_coeff[0] * slew_rate * rise_time."""
    rise_time = 0.002
    slew_rate = 150.0
    coeff = 0.8
    triangles = build_input_triangles(
        [rise_time], slew_rate=slew_rate, g_delay=0.0, enumerate_coeff=[coeff], dwell_time=1e-5, n_samples=2000
    )
    expected_amplitude = coeff * slew_rate * rise_time
    assert np.isclose(triangles.max(), expected_amplitude, rtol=1e-6)


def test_recovers_original_values_at_breakpoints():
    """Evaluating the PPoly at the original time points should reproduce the original amplitudes."""
    time = np.array([0.0, 1.0, 2.0, 3.0])
    amplitude = np.array([0.0, 2.0, -1.0, 0.0])
    gw_data = [np.stack((time, amplitude))]
    result = convert_waveforms_to_ppoly(gw_data)
    np.testing.assert_allclose(result[0](time), amplitude, atol=1e-10)


def test_linear_interpolation_between_breakpoints():
    """Midpoint between two breakpoints should be the linear average of their amplitudes."""
    time = np.array([0.0, 2.0])
    amplitude = np.array([0.0, 4.0])
    gw_data = [np.stack((time, amplitude))]
    result = convert_waveforms_to_ppoly(gw_data)
    assert np.isclose(result[0](1.0), 2.0, atol=1e-10)


def test_error_message_identifies_correct_channel_for_nan():
    """Test correct channel for invalid input"""
    gw_data = [
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        np.array([[0.0, 1.0], [0.0, np.nan]]),
    ]
    with pytest.raises(ValueError, match='channel 1'):
        convert_waveforms_to_ppoly(gw_data)


def test_error_message_identifies_correct_channel_for_inf():
    """Test correct channel for invalid input"""
    gw_data = [
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        np.array([[0.0, 1.0], [0.0, np.inf]]),
    ]
    with pytest.raises(ValueError, match='channel 2'):
        convert_waveforms_to_ppoly(gw_data)


def test_convert_waveforms_to_ppoly():
    """Test correct transformation to ppoly."""
    seq, _, _ = create_test_sequence()
    ppoly_gradients_pypulseq = seq.get_gradients()
    waveform_gradients = seq.waveforms()
    ppoly_gradients = convert_waveforms_to_ppoly(waveform_gradients)

    eval_points = np.linspace(0, 1, 100)
    assert ppoly_gradients[1] == ppoly_gradients_pypulseq[1]  # None
    assert ppoly_gradients[2] == ppoly_gradients_pypulseq[2]  # None
    np.testing.assert_allclose(ppoly_gradients[0](eval_points), ppoly_gradients_pypulseq[0](eval_points))


def test_calc_kspace_from_grad_waveforms():
    seq, _, _ = create_test_sequence()
    ppoly_gradients_pypulseq = seq.get_gradients()
    k_traj_adc_pp, k_traj_pp, _, _, _ = seq.calculate_kspace()
    k_traj_adc, k_traj, _, _, _ = calc_kspace_from_grad_waveforms(ppoly_gradients_pypulseq, seq)
    np.testing.assert_allclose(k_traj_adc, k_traj_adc_pp)
    np.testing.assert_allclose(k_traj, k_traj_pp)


def test_gmtf_construction():
    """Test GMTF constructor."""
    n = 50
    gmtf_x = np.random.rand(n) + 1j * np.random.rand(n)
    gmtf_y = np.random.rand(n) + 1j * np.random.rand(n)
    gmtf_z = np.random.rand(n) + 1j * np.random.rand(n)
    frequency = np.linspace(-1000, 1000, n)
    gmtf = Gmtf(gmtf_x=gmtf_x, gmtf_y=gmtf_y, gmtf_z=gmtf_z, frequency=frequency)
    assert gmtf.gmtf.shape == (3, n)
    np.testing.assert_array_equal(gmtf.gmtf[0], gmtf_x)
    np.testing.assert_array_equal(gmtf.gmtf[1], gmtf_y)
    np.testing.assert_array_equal(gmtf.gmtf[2], gmtf_z)


@pytest.mark.parametrize('scaling', (1, 0.8, 0.5))
@pytest.mark.parametrize('shift', (0, -2, 3))
def test_gmtf_trajectory_scaling(scaling, shift):
    """GMTF with different scaling and linear shifts."""
    n = 500
    seq, ktraj_orig, dwell_time = create_test_sequence()
    frequency = np.linspace(-1e5, 1e5, n)
    linear_phase = np.exp(-2j * np.pi * frequency * -shift * dwell_time)
    gmtf = Gmtf(gmtf_x=np.ones(n) * scaling * linear_phase, gmtf_y=np.ones(n), gmtf_z=np.ones(n), frequency=frequency)
    ktraj_gmtf = gmtf.correct_gradients(seq)
    np.testing.assert_allclose(ktraj_gmtf, ktraj_orig * scaling, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize('shift', (0, -2, 2))
def test_gmtf_trajectory_shift(shift):
    """GMTF with amplitude 1 and linear phase ramp."""
    n = 500
    seq, ktraj_orig, dwell_time = create_test_sequence()
    frequency = np.linspace(-1e5, 1e5, n)
    linear_phase = np.exp(-2j * np.pi * frequency * -shift * dwell_time)
    gmtf = Gmtf(gmtf_x=np.ones(n) * linear_phase, gmtf_y=np.ones(n), gmtf_z=np.ones(n), frequency=frequency)
    ktraj_gmtf = gmtf.correct_gradients(seq)
    np.testing.assert_allclose(ktraj_gmtf, ktraj_orig + shift, rtol=1e-3, atol=1e-2)
