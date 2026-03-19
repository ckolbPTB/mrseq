"""Receiver gain calibration for GRE and SE sequences."""

import numpy as np
import pypulseq as pp

from mrseq.utils import find_gx_flat_time_on_adc_raster
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults


def add_gre_receiver_gain_calibration(
    system: pp.Opts | None = None,
    seq: pp.Sequence | None = None,
    rf_flip_angle: float = 12.0,
    te: float = 4e-3,
    fov_z: float = 8e-3,
    fov_xy: float = 100e-3,
    n_readout: int = 128,
    adc_dwell_time: float = 24e-6,
):
    """Gradient echo receiver gain calibration.

    Gradient echo readout without phase encoding gradients for receiver gain calibration.

    Parameters
    ----------
    seq
        PyPulseq Sequence object.
    system
        PyPulseq system limit object.
    rf_flip_angle
        Flip angle of rf excitation pulse (in degrees)
    te
        Echo time (in seconds)
    fov_z
        Field of view in the z direction (slice thickness) (in meters).
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    adc_dwell_time
        ADC dwell time (in seconds).

    Returns
    -------
    seq
        PyPulseq Sequence object.
    block_duration
        Total duration of the reparation block (in seconds).
    """
    # set system to default if not provided
    if system is None:
        system = sys_defaults

    # create new sequence if not provided
    if seq is None:
        seq = pp.Sequence(system=system)

    minimum_time_to_set_label = round_to_raster(
        1e-5, system.block_duration_raster
    )  # minimum time to set a label (in seconds)

    # create slice selective excitation pulse and gradients
    rf_duration = round_to_raster(0.9e-3, system.rf_raster_time)
    rf_bwt = 2
    rf_apodization = 0.5
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=rf_flip_angle / 180 * np.pi,
        duration=rf_duration,
        slice_thickness=fov_z,
        apodization=rf_apodization,
        time_bw_product=rf_bwt,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
        use='excitation',
    )

    # create readout gradient and ADC
    gx_flat_time, adc_dwell_time = find_gx_flat_time_on_adc_raster(
        n_readout, adc_dwell_time, system.grad_raster_time, system.adc_raster_time
    )

    delta_k = 1 / fov_xy
    gx = pp.make_trapezoid(channel='x', flat_area=n_readout * delta_k, flat_time=gx_flat_time, system=system)
    adc = pp.make_adc(num_samples=n_readout, duration=gx.flat_time, delay=gx.rise_time, system=system)

    # create readout pre-winder
    gx_pre = pp.make_trapezoid(
        channel='x',
        area=-gx.area / 2 - delta_k / 2,
        system=system,
    )

    min_te = (
        rf.shape_dur / 2  # time from center to end of RF pulse
        + max(rf.ringdown_time, gz.fall_time)  # RF ringdown time or gradient fall time
        + pp.calc_duration(gzr, gx_pre)  # slice selection re-phasing gradient and readout pre-winder
        + gx.delay  # potential delay of readout gradient
        + gx.rise_time  # rise time of readout gradient
        + (n_readout // 2 + 0.5) * adc.dwell  # time from beginning of ADC to time point of k-space center sample
    )
    te_delay = round_to_raster(te - min_te, system.block_duration_raster)
    if te_delay < 0:
        raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')

    time_start = sum(seq.block_durations.values())
    seq.add_block(
        rf, gz, pp.make_label(type='SET', label='TRID', value=8888), pp.make_label(label='PMC', type='SET', value=1)
    )
    seq.add_block(gx_pre, gzr)
    seq.add_block(pp.make_delay(te_delay))
    seq.add_block(gx, adc)
    seq.add_block(pp.make_delay(minimum_time_to_set_label), pp.make_label(label='PMC', type='SET', value=0))
    block_duration = sum(seq.block_durations.values()) - time_start

    return seq, block_duration
