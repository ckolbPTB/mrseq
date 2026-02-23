"""Diffusion preparation block."""

import numpy as np
import pypulseq as pp

from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults


def add_diffusion_prep(
    seq: pp.Sequence | None = None,
    system: pp.Opts | None = None,
    fov_z: float = 0.008,
    rf_ex_duration: float = 1e-3,
    rf_ex_bwt: float = 4.0,
    rf_ref_duration: float = 2e-3,
    rf_ref_bwt: float = 4,
    rf_ref_width_scale_factor: float = 3.5,
    g_diff_amplitude: float | None = None,
    g_diff_duration: float = 1e-3,
    g_diff_delta_time: float | None = 5.5e-3,
) -> tuple[pp.Sequence, float, float, float]:
    """Add a diffusion preparation block to a sequence.

    The diffusion preparation block consists of a 90° excitation pulse, a diffusion gradient, a 180° refocusing
    pulse and finally a diffusion gradient again.

    Parameters
    ----------
    seq
        PyPulseq Sequence object.
    system
        PyPulseq system limit object.
    fov_z
        FOV along the slice direction.
    rf_ex_duration
        Duration of the excitation RF pulse in seconds.
    rf_ex_bwt
        Bandwidth-time product of the excitation RF pulse.
    rf_ref_duration
        Duration of the refocusing RF pulse in seconds.
    rf_ref_bwt
        Bandwidth-time product of the refocusing RF pulse.
    rf_ref_width_scale_factor
        Factor to scale the slice thickness of the refocusing pulse.
    g_diff_amplitude
        Amplitude of diffusion gradient. If set to None, 90% of the maximum amplitude is used.
    g_diff_duration
        Duration of diffusion gradient.
    g_diff_delta_time
        Time between beginning of first and second diffusion gradient. If None, the shortest possible time is used.

    Returns
    -------
    seq
        PyPulseq Sequence object.
    block_duration
        Total duration of the T1 preparation block (in seconds).
    te
        Echo time (in seconds).
    b_value
        b-value (in s/mm^2)

    """
    # set system to default if not provided
    if system is None:
        system = sys_defaults

    # create new sequence if not provided
    if seq is None:
        seq = pp.Sequence(system=system)

    # RF pulses
    rf_ex, gz_ex, gzr_ex = pp.make_sinc_pulse(
        flip_angle=np.pi / 2,
        duration=rf_ex_duration,
        slice_thickness=fov_z,
        apodization=0.5,
        phase_offset=0.0,
        time_bw_product=rf_ex_bwt,
        delay=system.rf_dead_time,  # delay should equal at least the dead time of the RF pulse
        system=system,
        return_gz=True,
        use='excitation',
    )

    rf_ref, gz_ref, _ = pp.make_sinc_pulse(
        flip_angle=np.pi,
        duration=rf_ref_duration,
        slice_thickness=fov_z * rf_ref_width_scale_factor,
        apodization=0.5,
        phase_offset=np.pi / 2,
        time_bw_product=rf_ref_bwt,
        delay=system.rf_dead_time,  # delay should equal at least the dead time of the RF pulse
        system=system,
        return_gz=True,
        use='refocusing',
    )

    # Diffusion gradient
    if g_diff_amplitude is None:
        g_diff_amplitude = system.max_grad * 0.9

    g_diff = [
        pp.make_trapezoid(channel=channel, system=system, amplitude=g_diff_amplitude, duration=g_diff_duration)
        for channel in ['x', 'y', 'z']
    ]
    g_diff = [
        pp.make_trapezoid(channel=channel, system=system, amplitude=g_diff_amplitude, duration=g_diff_duration)
        for channel in ['x']
    ]

    min_g_diff_delta_time = pp.calc_duration(*g_diff) + pp.calc_duration(rf_ref, gz_ref)
    if g_diff_delta_time is None:
        g_diff_delta_time = min_g_diff_delta_time
    if g_diff_delta_time < min_g_diff_delta_time:
        raise ValueError(
            f'Time delay between diffusion gradients must be larger than {min_g_diff_delta_time * 1000:.3f} ms. ',
            f'Current value is {g_diff_delta_time * 1000:.3f} ms.',
        )
    g_diff_delta_time_remaining = g_diff_delta_time - min_g_diff_delta_time

    # get current duration of sequence before adding T1 preparation block
    time_start = sum(seq.block_durations.values())

    # Add excitation pulse
    seq.add_block(rf_ex, gz_ex)
    seq.add_block(gzr_ex)
    seq.add_block(pp.make_delay(3e-3))

    # Add diffusion gradient
    start_of_first_diffusion_gradient = sum(seq.block_durations.values())
    seq.add_block(*g_diff)
    seq.add_block(pp.make_delay(round_to_raster(g_diff_delta_time_remaining / 2, system.block_duration_raster)))

    # Add refocusing pulse
    seq.add_block(rf_ref, gz_ref)

    # Add diffusion gradient
    seq.add_block(
        pp.make_delay(
            round_to_raster(
                g_diff_delta_time_remaining
                - round_to_raster(g_diff_delta_time_remaining / 2, system.block_duration_raster),
                system.block_duration_raster,
            )
        )
    )
    start_of_second_diffusion_gradient = sum(seq.block_durations.values())
    seq.add_block(*g_diff)

    # calculate total duration of diffusion block
    block_duration = sum(seq.block_durations.values()) - time_start

    time_between_excitation_refocusing = rf_ex.shape_dur / 2
    time_between_excitation_refocusing += max(rf_ex.ringdown_time, gz_ex.fall_time)
    time_between_excitation_refocusing += pp.calc_duration(gzr_ex)
    time_between_excitation_refocusing += pp.calc_duration(*g_diff)
    time_between_excitation_refocusing += max(rf_ref.delay, gz_ref.delay + gz_ref.rise_time)
    time_between_excitation_refocusing += rf_ref.shape_dur / 2

    g_diff_delta_time = start_of_second_diffusion_gradient - start_of_first_diffusion_gradient
    b_value = (
        (2 * np.pi * g_diff_amplitude) ** 2
        * g_diff_duration**2
        * (
            (g_diff_delta_time - g_diff_duration / 3)
            + g_diff[0].rise_time ** 3 / 30
            - g_diff_duration * g_diff[0].rise_time ** 2 / 6
        )
    )
    b_value *= 1e-6  # s/m^2 -> s/mm^2

    return (seq, block_duration, 2 * time_between_excitation_refocusing, b_value)
