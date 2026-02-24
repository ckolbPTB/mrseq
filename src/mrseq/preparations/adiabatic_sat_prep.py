"""Generate composite or adiabatic saturation pulse trains."""

import numpy as np
from pypulseq import Opts
from pypulseq import Sequence
from pypulseq import make_sinc_pulse
from pypulseq import make_trapezoid

from mrseq.utils.hyperbolic_secant_pulse import make_hypsec_90


def add_adia_sat_block(
    seq: Sequence | None = None,
    n_pulses: int = 3,
    max_b1: float = 20,
    sys: Opts | None = None,
) -> tuple[Sequence, float | None]:
    """Add adiabatic saturation pulse train to a PyPulseq Sequence object.

    Parameters
    ----------
    seq
        PyPulseq Sequence object
    n_pulses
        number of rf pulses in pulse train
    max_b1
        maximum B1 field amplitude in µT
    sys
        system limits

    Returns
    -------
    seq
        PyPulseq Sequence object
    last_spoil_dur
        duration of the last spoiler gradient in s
    """
    if not seq:
        seq = Sequence()

    if not sys:
        sys = Opts(max_grad=24, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s')

    # spoilers
    spoil_amp0 = 0.8 * sys.max_grad  # Hz/m
    spoil_amp1 = -0.7 * sys.max_grad  # Hz/m
    spoil_amp2 = 0.6 * sys.max_grad  # Hz/m

    rise_time = 1.0e-3  # spoiler rise time in seconds
    spoil_dur = 5.5e-3  # complete spoiler duration in seconds

    gx_spoil0, gy_spoil0, gz_spoil0 = (
        make_trapezoid(channel=c, system=sys, amplitude=spoil_amp0, duration=spoil_dur, rise_time=rise_time)
        for c in ['x', 'y', 'z']
    )
    gx_spoil1, gy_spoil1, gz_spoil1 = (
        make_trapezoid(channel=c, system=sys, amplitude=spoil_amp1, duration=spoil_dur, rise_time=rise_time)
        for c in ['x', 'y', 'z']
    )
    gx_spoil2, gy_spoil2, gz_spoil2 = (
        make_trapezoid(channel=c, system=sys, amplitude=spoil_amp2, duration=spoil_dur, rise_time=rise_time)
        for c in ['x', 'y', 'z']
    )

    hypsec_90 = make_hypsec_90(amp=max_b1, system=sys)

    for i in range(n_pulses):
        seq.add_block(hypsec_90)
        if i % 3 == 0:
            seq.add_block(gx_spoil0, gy_spoil1, gz_spoil2)
        elif i % 2 == 0:
            seq.add_block(gx_spoil2, gy_spoil0, gz_spoil1)
        else:
            seq.add_block(gx_spoil1, gy_spoil2, gz_spoil0)

    return seq, spoil_dur


def add_sat_pulse_train(
    seq: Sequence = None,
    n_pulses: int = 6,
    max_b1: float = 14,
    sys: Opts = None,
    grad_strength_factor: float = 1.0,
) -> tuple[Sequence, float | None]:
    """Create a saturation pulse train according to Chow, K. et al.

    For more information see:
    Saturation pulse design for quantitative myocardial T1 mapping. J. Cardiovasc. Magn. Reason. 17, 1-15 (2015).

    Parameters
    ----------
    seq
        PyPulseq Sequence object
    n_pulses
        number of rf pulses in pulse train
    max_b1
        maximum B1 field amplitude in µT
    sys
        system limits
    grad_strength_factor
        increase the spoiler gradient strength by this factor

    Returns
    -------
    seq
        PyPulseq Sequence object
    last_spoil_dur
        duration of the last spoiler gradient in s
    """
    if not seq:
        seq = Sequence()

    if not sys:
        sys = Opts(max_grad=24, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s')

    flip_angles = [115, 90, 125, 85, 176, 223]
    grad_dur = [5, 5.25, 4.5, 3.75, 3.5, 3, 4]
    grad_area_values = [153, 198, 136, 111, 72, 60, 119]
    grad_area = np.array(grad_area_values) * grad_strength_factor
    grad_ro = [-1, 1, -1, 0, 0, 0, 1]
    grad_pe = [0, 1, -1, 1, -1, 0, 0]
    grad_ss = [-1, 1, 0, -1, 0, 1, -1]

    dur_block = np.array(flip_angles) / (360 * sys.gamma * 10e-7 * max_b1)
    dur_sinc = np.ceil(dur_block / 0.22570566672775255 * 1e5) * 1e-5  # factor equals mean of sinc shape
    dur_block = np.ceil(dur_block * 1e5) * 1e-5

    for i in range(n_pulses + 1):
        if grad_ro[i] != 0:
            gx = make_trapezoid(
                channel='x', area=grad_ro[i] * grad_area[i], duration=grad_dur[i] / 1000, rise_time=5e-4
            )
        if grad_pe[i] != 0:
            gy = make_trapezoid(
                channel='y', area=grad_pe[i] * grad_area[i], duration=grad_dur[i] / 1000, rise_time=5e-4
            )
        if grad_ss[i] != 0:
            gz = make_trapezoid(
                channel='z', area=grad_ss[i] * grad_area[i], duration=grad_dur[i] / 1000, rise_time=5e-4
            )

        if grad_ro[i] != 0 and grad_pe[i] != 0 and grad_ss[i] != 0:
            seq.add_block(gx, gy, gz)
        elif grad_ro[i] != 0 and grad_pe[i] != 0:
            seq.add_block(gx, gy)
        elif grad_ro[i] != 0 and grad_ss[i] != 0:
            seq.add_block(gx, gz)
        elif grad_ro[i] != 0:
            seq.add_block(gx)
        elif grad_pe[i] != 0 and grad_ss[i] != 0:
            seq.add_block(gy, gz)
        elif grad_pe[i] != 0:
            seq.add_block(gy)
        elif grad_ss[i] != 0:
            seq.add_block(gz)

        if i < n_pulses:
            seq.add_block(make_sinc_pulse(flip_angle=flip_angles[i] * np.pi / 180, duration=dur_sinc[i], system=sys))

    return seq, grad_dur[-1] * 1e-3
