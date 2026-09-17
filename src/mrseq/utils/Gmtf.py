"""Gradient Modulation Transfer Function (GMTF) estimation and correction."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
import torch
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryCartesian
from pypulseq import eps
from scipy.interpolate import PPoly


def unwrap_phase_difference(data: torch.Tensor) -> torch.Tensor:
    """Return doubly-unwrapped phase of polarity[0] - polarity[1].

    Input shape:  ``(n_avg, 2, n_axes, n_rise, n_samples)``
    Output shape: ``(n_avg, n_axes, n_rise, n_samples)``
    """
    diff = data[:, 0, ...].angle() - data[:, 1, ...].angle()
    return torch.from_numpy(np.unwrap(np.unwrap(diff.numpy())))


def phase_to_gradient(
    phase_mean: torch.Tensor,
    phase_std: torch.Tensor,
    slice_pos: float,
    gamma: float,
    dwell_time: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert unwrapped-phase arrays to gradient waveforms via finite difference."""
    scale = slice_pos * gamma * dwell_time
    grad_mean = phase_mean.diff(dim=-1) / scale
    grad_std = phase_std.diff(dim=-1) / scale
    return grad_mean, grad_std


def build_input_triangles(
    rise_times: Sequence[float],
    slew_rate: float,
    g_delay: float,
    enumerate_coeff: Sequence[float],
    dwell_time: float,
    n_samples: int,
) -> torch.Tensor:
    """Build ideal triangular input waveforms.

    Returns
    -------
    Tensor of shape ``(n_rise, n_samples)`` [T/m].
    """
    triangles = torch.zeros(len(rise_times), n_samples)
    for i, rise_time in enumerate(rise_times):
        n_rise = round(rise_time / dwell_time)
        n_pre = round(g_delay / dwell_time)
        amplitude = enumerate_coeff[0] * slew_rate * rise_time

        slope_up = torch.linspace(0.0, amplitude, n_rise + 1)
        slope_down = torch.linspace(amplitude, 0.0, n_rise + 1)

        triangles[i, n_pre : n_pre + n_rise + 1] = slope_up
        triangles[i, n_pre + n_rise + 1 : n_pre + 2 * n_rise + 1] = slope_down[1:]

    return triangles


import torch


def apply_gmtf_to_single_gradient(
    input_gradient: torch.Tensor, input_gradient_time: torch.Tensor, gmtf: torch.Tensor, gmtf_frequency: torch.Tensor
) -> torch.Tensor:
    """
    Apply a Gradient Modulation Transfer Function (GMTF) correction.

    The input gradient is resampled onto a uniform time grid matching the GMTF's frequency resolution, transformed
    to the frequency domain, multiplied by the GMTF, and transformed back to obtain the predicted (corrected)
    gradient waveform.

    Parameters
    ----------
    input_gradient
        Nominal gradient waveform, shape (M,).
    input_gradient_time
        1D tensor of time points (s) corresponding to `input_gradient`, of length M.
    gmtf
        Complex-valued GMTF, shape (N,).
    gmtf_frequency
        1D tensor of frequency values (Hz) corresponding to `gmtf`, of length N.

    Returns
    -------
        Corrected gradient waveform, shape (3, N // 2), real-valued. (The result is theoretically complex, but
        since the GMTF spectrum is symmetric, the imaginary part is expected to be negligible and is discarded.)
    """
    if gmtf.shape[-1] != len(gmtf_frequency):
        raise ValueError('Gmtf values and frequency information need to have the same length.')
    if input_gradient.shape[-1] != len(input_gradient_time):
        raise ValueError('Input gradient and time information need to have the same length.')

    n_freq = gmtf_frequency.shape[0]
    n_time = n_freq // 2

    # Time step and time axis matching the GMTF's frequency resolution
    dt = 1 / gmtf_frequency[-1] / 2
    girf_time = dt * torch.arange(n_time)

    # Resample the nominal gradient onto the GMTF's time grid
    grad_interp = torch.as_tensor(np.interp(girf_time, input_gradient_time, input_gradient, left=0, right=0))

    # Forward FFT (zero-padded to 2x length to match GMTF's frequency grid)
    grad_spectrum = torch.fft.fftshift(torch.fft.fft(grad_interp, n=2 * grad_interp.shape[-1], dim=-1), dim=-1)

    # Apply GMTF correction in the frequency domain
    corrected_spectrum = torch.fft.ifftshift(gmtf * grad_spectrum, dim=-1)

    # Inverse FFT back to time domain, keep only the non-padded part
    corrected_gradient = torch.fft.ifft(corrected_spectrum, dim=-1)
    corrected_gradient = corrected_gradient[: grad_interp.shape[-1]]

    # The corrected waveform is technically complex; since the GMTF spectrum is symmetric, the imaginary part
    # should be very close to zero.
    return torch.real(corrected_gradient), girf_time[: grad_interp.shape[-1]]


def apply_gmtf_to_sequence(seq_file: str | Path, gmtf: 'Gmtf') -> list[np.ndarray]:
    """
    Apply Gmtf correction to all gradient waveforms in sequence and return corrected gradient waveform.

    Parameters
    ----------
        seq_file
            Path to the Pulseq (.seq) file describing the nominal gradient waveforms that were played out during the
            measurement.
        gmtf:
            Gmtf instance

    Returns
    -------
        Corrected gradient waveform
    """
    seq = pp.Sequence()
    seq.read(str(seq_file))

    # get times
    t_excitation = seq.rf_times()[0]
    t_end = sum(seq.block_durations.values())

    # Read waveforms from Sequence and directly split into TR blocks.
    # The results will be stored in gw_blocks, which is a list of len(t_excitation). Each entry is a list of
    # length 3 (for x,y,z gradients). Each of the 3 entries is a np.array with 2 rows (time, amplitude).
    gradient_waveform_input_blocks = []
    for n, t_exc in enumerate(t_excitation):
        block_start = t_exc
        block_end = t_end if n == len(t_excitation) - 1 else t_excitation[n + 1]
        gw_block = seq.waveforms(time_range=(block_start, block_end))
        gradient_waveform_input_blocks.append(gw_block)

    gradient_waveform_corrected_blocks = []
    for tr_idx, gw in enumerate(gradient_waveform_input_blocks):
        gradient_waveform_corrected_blocks.append([])
        for grad_idx, grad in enumerate(gw):
            if len(grad[1]):
                # subtract start time of the TR block from the time vector
                grad_input_time = grad[0] - grad[0][0]
                grad_corrected, grad_corrected_time = apply_gmtf_to_single_gradient(
                    grad[1], grad_input_time, gmtf.gmtf[grad_idx, :], gmtf.frequency
                )
                gradient_waveform_corrected_blocks[tr_idx].append((grad_corrected_time + grad[0][0], grad_corrected))
            else:
                gradient_waveform_corrected_blocks[tr_idx].append(([], []))

    return gradient_waveform_corrected_blocks


def convert_waveforms_to_ppoly(gw_data: list[np.ndarray]) -> list[PPoly]:
    """
    Convert gradient waveforms into piecewise polynomial (PPoly) representations.

    Each waveform is padded with tiny zero-amplitude segments just before its start and after its end (to ensure clean
    extrapolation to zero outside the defined time range), and then converted into a linear piecewise polynomial
    using the time points as breakpoints.

    Parameters
    ----------
    gw_data
        List of gradient waveforms, one per gradient channel. Each waveform is expected to have shape (2, M): the first
        row containing time points (s) and the second row containing the corresponding gradient amplitudes.

    Returns
    -------
    List of piecewise polynomial representations, one per gradient channel, in the same order as `gw_data`.

    Raises
    ------
    ValueError
        If any waveform contains non-finite values (NaN or Inf).
    """
    n_grad_channels = len(gw_data)
    eps = 1e-12

    gw_pp = []
    for grad_idx in range(n_grad_channels):
        gw = gw_data[grad_idx]

        if not np.all(np.isfinite(gw)):
            raise ValueError(f'Gradient channel {grad_idx}: not all elements of the generated waveform are finite.')

        # Pad with near-zero-amplitude points just before/after the waveform
        # so that extrapolation outside the defined range goes cleanly to zero.
        pre_pad = np.array([[gw[0, 0] - 2 * eps, gw[0, 0] - eps], [0, 0]])
        post_pad = np.array([[gw[0, -1] + eps, gw[0, -1] + 2 * eps], [0, 0]])
        gw = np.hstack((pre_pad, gw, post_pad))

        # Avoid signed-zero artifacts (-0.0) in the amplitude row
        gw[1][gw[1] == -0.0] = 0.0

        time = gw[0]
        amplitude = gw[1]
        slopes = np.diff(amplitude) / np.diff(time)

        gw_pp.append(PPoly(np.stack((slopes, amplitude[:-1])), time, extrapolate=True))

    return gw_pp


def calc_kspace_from_grad_waveforms(gw_pp, seq):
    # get timings from sequence
    total_duration = sum(seq.block_durations.values())
    t_excitation, fp_excitation, t_refocusing, _ = seq.rf_times()
    t_adc, _ = seq.adc_times()

    ng = len(gw_pp)

    # Calculate slice positions.
    # For now we entirely rely on the excitation -- ignoring complicated interleaved refocused sequences
    if len(t_excitation) > 0:
        # Position in x, y, z
        slice_pos = np.zeros((ng, len(t_excitation)))
        for j in range(ng):
            if gw_pp[j] is None:
                slice_pos[j] = np.nan
            else:
                # Check for divisions by zero to avoid numpy warning
                divisor = np.array(gw_pp[j](t_excitation))
                slice_pos[j, divisor != 0.0] = fp_excitation[0, divisor != 0.0] / divisor[divisor != 0.0]
                slice_pos[j, divisor == 0.0] = np.nan

        slice_pos[~np.isfinite(slice_pos)] = 0  # Reset undefined to 0
    else:
        slice_pos = []

    # Integrate waveforms as PPs to produce gradient moments
    gm_pp = []
    tc = []
    for i in range(ng):
        if gw_pp[i] is None:
            gm_pp.append(None)
            continue

        gm_pp.append(gw_pp[i].antiderivative())
        tc.append(gm_pp[i].x)
        # "Sample" ramps for display purposes.  Otherwise piecewise-linear display (plot) fails
        ii = np.flatnonzero(np.abs(gm_pp[i].c[0, :]) > 1e-7 * seq.system.max_slew)

        # Do nothing if there are no ramps
        if ii.shape[0] == 0:
            continue

        starts = np.int64(np.floor((gm_pp[i].x[ii] + eps) / seq.grad_raster_time))
        ends = np.int64(np.ceil((gm_pp[i].x[ii + 1] - eps) / seq.grad_raster_time))

        # Create all ranges starts[0]:ends[0], starts[1]:ends[1], etc.
        lengths = ends - starts + 1
        inds = np.ones((lengths).sum())
        # Calculate output index where each range will start
        start_inds = np.cumsum(np.concatenate(([0], lengths[:-1])))
        # Create element-wise differences that will cumsum into
        # the final indices: [starts[0], 1, 1, starts[1]-starts[0]-lengths[0]+1, 1, etc.]
        inds[start_inds] = np.concatenate(([starts[0]], np.diff(starts) - lengths[:-1] + 1))

        tc.append(np.cumsum(inds) * seq.grad_raster_time)
    if tc != []:
        tc = np.concatenate(tc)

    t_acc = 1e-10  # Temporal accuracy
    t_acc_inv = 1 / t_acc
    # tc = self.__flatten_jagged_arr(tc)
    t_ktraj = t_acc * np.unique(
        np.round(
            t_acc_inv
            * np.array(
                [
                    *tc,
                    0,
                    *np.asarray(t_excitation) - 2 * seq.rf_raster_time,
                    *np.asarray(t_excitation) - seq.rf_raster_time,
                    *t_excitation,
                    *np.asarray(t_refocusing) - seq.rf_raster_time,
                    *t_refocusing,
                    *t_adc,
                    total_duration,
                ]
            )
        )
    )

    i_excitation = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_excitation)))
    i_refocusing = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_refocusing)))
    i_adc = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_adc)))

    i_periods = np.unique([0, *i_excitation, *i_refocusing, len(t_ktraj) - 1])
    if len(i_excitation) > 0:
        ii_next_excitation = 0
    else:
        ii_next_excitation = -1
    if len(i_refocusing) > 0:
        ii_next_refocusing = 0
    else:
        ii_next_refocusing = -1

    k_traj = np.zeros((ng, len(t_ktraj)))
    for i in range(ng):
        if gw_pp[i] is None:
            continue

        it = np.where(
            np.logical_and(
                t_ktraj >= t_acc * round(t_acc_inv * gm_pp[i].x[0]),
                t_ktraj <= t_acc * round(t_acc_inv * gm_pp[i].x[-1]),
            )
        )[0]
        k_traj[i, it] = gm_pp[i](t_ktraj[it])
        if t_ktraj[it[-1]] < t_ktraj[-1]:
            k_traj[i, it[-1] + 1 :] = k_traj[i, it[-1]]

    # Convert gradient moments to k-space positions
    dk = -k_traj[:, 0]
    for i in range(len(i_periods) - 1):
        i_period = i_periods[i]
        i_period_end = i_periods[i + 1]
        if ii_next_excitation >= 0 and i_excitation[ii_next_excitation] == i_period:
            if abs(t_ktraj[i_period] - t_excitation[ii_next_excitation]) > t_acc:
                raise Warning(
                    f'abs(t_ktraj[i_period]-t_excitation[ii_next_excitation]) < {t_acc} failed for ii_next_excitation={ii_next_excitation} error={t_ktraj(i_period) - t_excitation(ii_next_excitation)}'
                )
            dk = -k_traj[:, i_period]
            if i_period > 0:
                # Use nans to mark the excitation points since they interrupt the plots
                k_traj[:, i_period - 1] = np.nan
            # -1 on len(i_excitation) for 0-based indexing
            ii_next_excitation = min(len(i_excitation) - 1, ii_next_excitation + 1)
        elif ii_next_refocusing >= 0 and i_refocusing[ii_next_refocusing] == i_period:
            # dk = -k_traj[:, i_period]
            dk = -2 * k_traj[:, i_period] - dk
            # -1 on len(i_excitation) for 0-based indexing
            ii_next_refocusing = min(len(i_refocusing) - 1, ii_next_refocusing + 1)

        k_traj[:, i_period:i_period_end] = k_traj[:, i_period:i_period_end] + dk[:, None]

    k_traj[:, i_period_end] = k_traj[:, i_period_end] + dk
    k_traj_adc = k_traj[:, i_adc]

    return k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc


def estimate_gmtf(
    grad_input: torch.Tensor,
    grad_output_mean: torch.Tensor,
) -> torch.Tensor:
    """Least-squares GMTF estimate over all rise times.

    Parameters
    ----------
    grad_input:
        Ideal waveforms, shape ``(n_rise, n_samples)``.
    grad_output_mean:
        Measured waveforms, shape ``(n_axes, n_rise, n_samples-1)``.

    Returns
    -------
    Complex tensor of shape ``(n_axes, n_freq)``.
    """
    n_fft_in = 2 * (grad_input.shape[-1])
    n_fft_out = 2 * grad_output_mean.shape[-1]

    in_spec = torch.fft.fftshift(torch.fft.fft(grad_input, n=n_fft_in, dim=-1), dim=-1)
    out_spec = torch.fft.fftshift(torch.fft.fft(grad_output_mean, n=n_fft_out, dim=-1), dim=-1)

    # Least-squares: sum_rt conj(H_in) * H_out  /  sum_rt |H_in|^2
    numerator = (in_spec.conj() * out_spec).sum(dim=-2)  # sum over rise dim
    denominator = (in_spec.abs() ** 2).sum(dim=-2)
    return numerator / denominator


class Gmtf:
    """
    Gradient Modulation Transfer Function (GMTF) container and utilities.

    The GMTF describes the frequency-domain response of the gradient system for each physical axis (x, y, z), and is
    related to the Gradient Impulse Response Function (GIRF) via GMTF = FT(GIRF).

    Attributes
    ----------
    gmtf
        Complex-valued tensor containing the concatenated GMTF for the x, y, and z axes (in that order),
        with shape (3, N), where N is the number of frequency samples.
    frequency
        1D tensor of frequency values (Hz) corresponding to the samples in `gmtf`, of length N.
    """

    def __init__(
        self, gmtf_z: torch.Tensor, gmtf_y: torch.Tensor, gmtf_x: torch.Tensor, frequency: torch.Tensor
    ) -> None:
        """
        Initialize the GMTF container.

        Parameters
        ----------
        gmtf_x
            Complex-valued 1D tensor containing the GMTF for the x-axis, of length N.
        gmtf_y
            Complex-valued 1D tensor containing the GMTF for the y-axis, of length N.
        gmtf_z
            Complex-valued 1D tensor containing the GMTF for the z-axis, of length N.
        frequency
            1D real-valued tensor of frequency values (Hz), of length N, corresponding to the frequency axis of
            `gmtf_x`, `gmtf_y`, and`gmtf_z`.

        Raises
        ------
        ValueError
            If `gmtf_x`, `gmtf_y`, `gmtf_z`, and `freq` are not all 1D tensors of the same length.

        Notes
        -----
        Sets `self.gmtf` as the concatenation (stacking) of `gmtf_x`, `gmtf_y`, and `gmtf_z`, in that order,
        along a new leading dimension, resulting in shape (3, N).
        """
        lengths = {
            'gmtf_x': gmtf_z.shape[0],
            'gmtf_y': gmtf_y.shape[0],
            'gmtf_z': gmtf_x.shape[0],
            'freq': frequency.shape[0],
        }
        if len(set(lengths.values())) > 1:
            raise ValueError(f'gmtf_x, gmtf_y, gmtf_z, and freq must all have the same length, got: {lengths}.')

        self.gmtf = torch.stack((gmtf_x, gmtf_y, gmtf_z))
        self.frequency = frequency

    @classmethod
    def compute_gmtf(cls, mrd_file: str | Path, seq_file: str | Path) -> 'Gmtf':
        """
        Compute the GMTF from a measured MRD dataset and the corresponding Pulseq sequence file used to acquire it.

        Parameters
        ----------
        mrd_file
            Path to the measured raw data file in MRD (.mrd/.h5) format, containing the measured gradient response
            (e.g. field-camera or phantom-based measurement) for each axis.
        seq_file
            Path to the Pulseq (.seq) file describing the nominal gradient waveforms that were played out during the
            measurement.

        Returns
        -------
        Gmtf
            A new GMTF instance containing the computed gradient modulation transfer function and associated frequency
            vector for the x, y, and z axes.
        """
        # Load k-space data, sort into correct dimensions and combine to single coil
        kdata = KData.from_file(mrd_file, trajectory=KTrajectoryCartesian())
        idx = np.lexsort(
            (
                kdata.header.acq_info.idx.phase.squeeze(),
                kdata.header.acq_info.idx.repetition.squeeze(),
                kdata.header.acq_info.idx.average.squeeze(),
            )
        )
        kdata_sorted = kdata[torch.as_tensor(idx)]
        kdata_sorted = kdata_sorted.rearrange(
            '(avg rep ph) ... -> avg rep ph ...',
            rep=int(kdata.header.acq_info.idx.repetition.max()) + 1,
            avg=int(kdata.header.acq_info.idx.average.max()) + 1,
            ph=int(kdata.header.acq_info.idx.phase.max()) + 1,
        )
        kdata_single_coil = kdata_sorted.compress_coils(n_compressed_coils=1).data.squeeze()

        # Get additional information from sequence
        sequence = pp.Sequence()
        sequence.read(seq_file)

        dwell_time = sequence.get_definition('DwellTime')
        slice_pos = sequence.get_definition('SlicePos')
        gamma = sequence.system.gamma * 2 * torch.pi
        rise_times = sequence.get_definition('RiseTimes')
        slew_rate = sequence.get_definition('SlewRate') / sequence.system.gamma
        g_delay = sequence.get_definition('GradientPreEmphasisDelay')
        g_amplitude_coeff = sequence.get_definition('GradAmplitudeCoeff')

        # Unwrape phase
        phase = unwrap_phase_difference(kdata_single_coil)
        phase_mean = phase.mean(dim=0)
        phase_std = phase.std(dim=0)
        grad_output_mean, _grad_output_std = phase_to_gradient(phase_mean, phase_std, slice_pos, gamma, dwell_time)

        # Calculate ideal triangles
        grad_input = build_input_triangles(
            rise_times, slew_rate, g_delay, g_amplitude_coeff, dwell_time, grad_output_mean.shape[-1]
        )

        # Check for any sign flips
        corr = torch.sum(grad_input * grad_output_mean)
        grad_output_mean = grad_output_mean if corr >= 0 else grad_output_mean * -1

        gmtf = estimate_gmtf(grad_input, grad_output_mean)
        frequency = (1.0 / dwell_time) * torch.arange(-gmtf.shape[-1] // 2, gmtf.shape[-1] // 2) / (gmtf.shape[-1])
        return Gmtf(gmtf_x=gmtf[0, :], gmtf_y=gmtf[1, :], gmtf_z=gmtf[2, :], frequency=frequency)

    def plot(
        self,
        frequency_lim: tuple[float, float] | None = None,
        amplitude_lim: tuple[float, float] | None = None,
        phase_lim: tuple[float, float] | None = None,
    ) -> None:
        """
        Plot the magnitude and phase of the GMTF for each gradient axis.

        Parameters
        ----------
        frequency_lim
            (min, max) frequency range (Hz) to display on the x-axis of both subplots. If None, the full frequency
            range is shown.
        amplitude_lim
            (min, max) limits for the amplitude/magnitude subplot y-axis. If None, limits are chosen automatically.
        phase_lim
            (min, max) limits for the phase subplot y-axis (in radians). If None, limits are chosen automatically.

        """
        if frequency_lim is None:
            frequency_lim = (0.0, 10.0)
        if amplitude_lim is None:
            amplitude_lim = (0.8, 1.025)
        if phase_lim is None:
            phase_lim = (-0.1, np.pi / 4)

        # Plot magnitude and phase of the GMTF for all axes
        grad_axes = ['x', 'y', 'z']

        half = len(self.frequency) // 2

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
        for k, label in enumerate(grad_axes):
            ax1.plot(self.frequency[half:] * 1e-3, abs(self.gmtf.numpy()[k, half:]), label=label)
            ax2.plot(self.frequency[half:] * 1e-3, np.unwrap(np.angle(self.gmtf.numpy()[k, half:])), label=label)

        ax1.set(xlabel='Frequency (kHz)', title='GMTF Magnitude', xlim=frequency_lim, ylim=amplitude_lim)
        ax2.set(xlabel='Frequency (kHz)', title='GMTF Phase', xlim=frequency_lim, ylim=phase_lim)

        for ax in (ax1, ax2):
            ax.legend()
            ax.grid(visible=True, which='major', color='#666666')
            ax.minorticks_on()
            ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

        fig.tight_layout()
        plt.show()

    def correct_gradients(
        self,
        seq_file: str | Path,
        freq_threshold: float,
    ) -> torch.Tensor:
        """
        Apply GMTF-based correction to the gradient waveforms in a Pulseq sequence file, and output new trajectory.

        Parameters
        ----------
        seq_file
            Path to the input Pulseq (.seq) file whose gradient waveform (x, y, z) should be corrected.
        freq_threshold
            Frequency (Hz) above which the GMTF-based correction is not applied (e.g. to avoid amplifying noise or
            unreliable GMTF estimates at high frequencies).

        Returns
        -------
            Corrected trajectory.
        """
