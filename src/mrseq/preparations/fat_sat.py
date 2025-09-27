"""Fat saturation block."""

import numpy as np
import pypulseq as pp

from mrseq.utils import sys_defaults


def add_fat_sat(
    seq: pp.Sequence | None = None,
    system: pp.Opts | None = None,
    rf_duration: float = 8e-3,
    rf_flip_angle: float = 110,
    saturation_frequency_ppm=-3.45,
) -> tuple[pp.Sequence, float, float]:
    """Add fat saturation pulse.

    This code was adapted from
    https://github.com/imr-framework/pypulseq/blob/master/examples/scripts/write_epi_se_rs.py

    seq
        PyPulseq Sequence object.
    system
        PyPulseq system limit object.
    rf_duration
        Duration of the adiabatic inversion pulse (in seconds).
    rf_flip_angle
        Flip angle of the fat saturation pulse (in degrees).
    saturation_frequency_ppm
        Frequency offset of the fat saturation pulse (in ppm).

    Returns
    -------
    seq
        PyPulseq Sequence object.
    block_duration
        Total duration of the fat saturation block (in seconds).
    """
    # set system to default if not provided
    if system is None:
        system = sys_defaults

    if system.B0 is None:
        raise ValueError('B0 field strength must be defined in system settings.')

    # create new sequence if not provided
    if seq is None:
        seq = pp.Sequence(system=system)

    # get current duration of sequence before adding T2 preparation block
    time_start = sum(seq.block_durations.values())

    sat_freq = saturation_frequency_ppm * 1e-6 * system.B0 * system.gamma
    rf_fs = pp.make_gauss_pulse(
        flip_angle=np.deg2rad(rf_flip_angle),
        system=system,
        duration=rf_duration,
        delay=system.rf_dead_time,
        bandwidth=np.abs(sat_freq),
        freq_offset=sat_freq,
        use='preparation',
    )
    gz_fs = pp.make_trapezoid(
        channel='z',
        system=system,
        delay=pp.calc_duration(rf_fs),
        area=1 / 1e-4,  # spoil up to 0.1mm
    )

    seq.add_block(rf_fs, gz_fs)

    # calculate total block duration
    block_duration = sum(seq.block_durations.values()) - time_start

    return (seq, block_duration)
