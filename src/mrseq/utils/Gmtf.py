"""Gradient Modulation Transfer Function (GMTF) estimation and correction."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
import torch
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryCartesian


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
        Complex-valued tensor containing the concatenated GMTF for the z, y, and x axes (in that order),
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
        gmtf_z
            Complex-valued 1D tensor containing the GMTF for the z-axis, of length N.
        gmtf_y
            Complex-valued 1D tensor containing the GMTF for the y-axis, of length N.
        gmtf_x
            Complex-valued 1D tensor containing the GMTF for the x-axis, of length N.
        frequency
            1D real-valued tensor of frequency values (Hz), of length N, corresponding to the frequency axis of
            `gmtf_z`, `gmtf_y`, and`gmtf_x`.

        Raises
        ------
        ValueError
            If `gmtf_z`, `gmtf_y`, `gmtf_x`, and `freq` are not all 1D tensors of the same length.

        Notes
        -----
        Sets `self.gmtf` as the concatenation (stacking) of `gmtf_z`, `gmtf_y`, and `gmtf_x`, in that order,
        along a new leading dimension, resulting in shape (3, N).
        """
        lengths = {
            'gmtf_z': gmtf_z.shape[0],
            'gmtf_y': gmtf_y.shape[0],
            'gmtf_x': gmtf_x.shape[0],
            'freq': frequency.shape[0],
        }
        if len(set(lengths.values())) > 1:
            raise ValueError(f'gmtf_z, gmtf_y, gmtf_x, and freq must all have the same length, got: {lengths}.')

        self.gmtf = torch.stack((gmtf_z, gmtf_y, gmtf_x))
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
            vector for the z, y, and x axes.
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
        return Gmtf(gmtf_z=gmtf[-1, :], gmtf_y=gmtf[-2, :], gmtf_x=gmtf[-3, :], frequency=frequency)

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
        grad_axes = ['z', 'y', 'x']

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
