"""Diffusion preparation block."""

from typing import Literal

import numpy as np
import pypulseq as pp

from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults


def calculate_b_value(
    g_amplitude: float,
    g_duration: float,
    g_rise_time: float,
    g_delta_time: float,
) -> float:
    """
    Calculate b-values.

    Calculate the b-value for diffusion-weighted imaging using two
    diffusion gradients (trapezoidal) separated by refocusing pulses.

    Parameters
    ----------
    g_amplitude
        Gradient amplitude for diffusion weighting in Hz/m.
    g_duration
        Gradient duration for diffusion weighting in seconds.
    g_rise_time
        Gradient rise time for diffusion weighting in seconds.
    g_delta_time
        Gap between diffusion gradients in seconds.

    Returns
    -------
    b_value
        The calculated b-value in s/mm^2.
    """
    b_value = (
        (2 * np.pi) ** 2
        * g_amplitude**2
        * (g_duration**2 * (g_delta_time - g_duration / 3) + g_rise_time**3 / 30 - g_duration * g_rise_time**2 / 6)
    )
    b_value = b_value * 1e-6  # Convert from s/m^2 to s/mm^2

    return b_value


def calculate_diffusion_gradient_duration(b_value: float, g_amplitude: float, g_delta_time: float) -> float:
    """
    Calculate diffusion gradient duration.

    Calculate the duration of diffusion gradients based on the b-value,
    gradient amplitude and duration between both gradients.
    The calculation assumes a negligible rise time in comparison to the plateau duration,
    such that the equation for rectangular gradients can be applied.

    b = (2 * np.pi)^2 * g_amplitude^2 * g_duration^2 * (g_delta_time - g_duration/3)

    which cen be reformulated to a cubic equation

    -(2 * np.pi)^2 * g_amplitude^2/3 * g_duration^3 + (2 * np.pi)^2 * g_amplitude^2 * g_delta_time * g_duration^2 - b = 0

    The solution for this can be found with np.roots

    Parameters
    ----------
    b_value
        The b-value in s/mm^2.
    amp
        Gradient amplitude for diffusion weighting in Hz/m.
    g_delta_time
        Gap between diffusion gradients in seconds.

    Returns
    -------
    g_duration
        The calculated gradient duration in seconds.
    """
    # Convert b value from s/mm^2 to s/m^2
    coeffs = [
        -((2 * np.pi) ** 2) * g_amplitude**2 / 3,
        (2 * np.pi) ** 2 * g_amplitude**2 * g_delta_time,
        0,
        -b_value * 1e6,
    ]
    roots = np.roots(coeffs)
    # Only keep the real, positive root which is smaller than the g_delta_time
    g_duration = [r.real for r in roots if np.isreal(r) and 0 < r.real < g_delta_time]
    if len(g_duration) == 0:
        raise ValueError('No valid duration found for given parameters')
    return g_duration[0].item()


class DiffusionPrep:
    """
    Multi-echo gradient echo acquisition.

    Attributes
    ----------
    system
        PyPulseq system limits object.
    rf_ref
        180° refocusing pulse
    gz_ref
        Slice encoding gradient for the refocusing pulse
    g_diff
        Diffusion gradient.
    g_delta_time
        Gap between diffusion gradients in seconds.
    max_b_value
        Highest b-value in s/mm^2
    """

    def __init__(
        self,
        system: pp.Opts | None = None,
        fov_z: float = 0.008,
        rf_ref_duration: float = 2e-3,
        rf_ref_bwt: float = 4,
        rf_ref_width_scale_factor: float = 3.5,
        g_amplitude: float | None = None,
        max_b_value: float = 500e-3,
        g_delta_time: float = 10e-3,
        g_channel: Literal['x', 'y', 'z'] = 'x',
        g_crusher_duration=1.6e-3,
    ):
        """
        Initialize the DiffusionPrep class and compute all required attributes.

        Parameters
        ----------
        system
        PyPulseq system limit object.
        fov_z
            FOV along the slice direction.
        rf_ref_duration
            Duration of the refocusing RF pulse in seconds.
        rf_ref_bwt
            Bandwidth-time product of the refocusing RF pulse.
        rf_ref_width_scale_factor
            Factor to scale the slice thickness of the refocusing pulse.
        g_amplitude
            Amplitude of diffusion gradient. If set to None, 90% of the maximum amplitude is used.
        max_b_value
            Highest b-value in s/mm^2
        g_delta_time
            Time between beginning of first and second diffusion gradient. If None, the shortest possible time is used.
        g_channel
            Axis of the diffusion gradient
        g_crusher_duration
            Duration of crusher gradient for b-value = 0
        """
        # set system to default if not provided
        self.system_ = system if system else sys_defaults

        # Diffusion gradient
        if g_amplitude is None:
            g_amplitude = self.system_.max_grad * 0.9

        # Calculate duration based on highest b-value
        g_duration = calculate_diffusion_gradient_duration(max_b_value, g_amplitude, g_delta_time)

        self.g_diff_ = pp.make_trapezoid(
            channel=g_channel, system=self.system_, amplitude=g_amplitude, duration=g_duration
        )

        # Refousing pulse
        self._rf_ref, self._gz_ref, _ = pp.make_sinc_pulse(
            flip_angle=np.pi,
            duration=rf_ref_duration,
            slice_thickness=fov_z * rf_ref_width_scale_factor,
            apodization=0.5,
            phase_offset=np.pi / 2,
            time_bw_product=rf_ref_bwt,
            delay=self.system_.rf_dead_time,
            system=self.system_,
            return_gz=True,
            use='refocusing',
        )

        self.max_b_value_ = max_b_value

        # Crusher gradient for b_value == 0
        self.g_crush_ = pp.make_trapezoid(
            channel=g_channel, system=self.system_, amplitude=g_amplitude, duration=g_crusher_duration
        )

        # Calculate timings
        self.g_delta_time_ = g_delta_time
        min_g_delta_time = pp.calc_duration(self.g_diff_) + pp.calc_duration(self.rf_ref_, self.gz_ref_)
        self.time_between_diff_and_rf_ = round_to_raster(
            (g_delta_time - min_g_delta_time) / 2, raster_time=self.system_.block_duration_raster
        )
        self.time_between_rf_and_diff_ = round_to_raster(
            (g_delta_time - min_g_delta_time) - self.time_between_diff_and_rf,
            raster_time=self.system_.block_duration_raster,
        )
        self.delay_before_crush_ = round_to_raster(
            pp.calc_duration(self.g_diff_) - pp.calc_duration(self.g_crush_) + self.time_between_diff_and_rf,
            raster_time=self.system_.block_duration_raster,
        )
        self.delay_after_crush_ = round_to_raster(
            pp.calc_duration(self.g_diff_) - pp.calc_duration(self.g_crush_) + self.time_between_rf_and_diff,
            raster_time=self.system_.block_duration_raster,
        )

        self.time_to_refocusing_pulse_ = pp.calc_duration(self.g_diff_)
        self.time_to_refocusing_pulse_ += max(self.rf_ref_.delay, self.gz_ref_.delay + self.gz_ref_.rise_time)
        self.time_to_refocusing_pulse_ += self.rf_ref_.shape_dur / 2

    def add_diffusion_prep(self, seq: pp.Sequence | None, b_value: float):
        """Add a diffusion preparation block to a sequence.

        The diffusion preparation block consists of a diffusion gradient, a 180° refocusing
        pulse and another diffusion gradient.

        Parameters
        ----------
        seq
            PyPulseq Sequence object.
        b_value
            Current b-value in s/mm^2
        """
        if b_value > self.max_b_value_:
            raise ValueError(f'Current b-value {b_value} larger than highest value {self.max_b_value_}')

        # create new sequence if not provided
        if seq is None:
            seq = pp.Sequence(system=self.system_)

        time_start = sum(seq.block_durations.values())

        # Add diffusion gradient
        if b_value > 0:
            g_current_diff = pp.scale_grad(self.g_diff_, np.sqrt(b_value / self.max_b_value_))
            seq.add_block(g_current_diff)
            seq.add_block(self.time_between_diff_and_rf_)
        else:
            seq.add_block(self.delay_before_crush_)
            seq.add_block(self.g_crush_)

        # Add refocusing pulse
        seq.add_block(self.rf_ref_, self.gsz_ref_)

        if b_value > 0:
            seq.add_block(self.time_between_rf_and_diff_)
            seq.add_block(g_current_diff)
        else:
            seq.add_block(self.g_crush_)
            seq.add_block(self.delay_after_crush_)

        # calculate total duration of diffusion block
        block_duration = sum(seq.block_durations.values()) - time_start

        b_value = calculate_b_value(
            g_current_diff.amplitude, g_current_diff.duration, g_current_diff.rise_time, self.g_delta_time_
        )

        return seq, block_duration, b_value
