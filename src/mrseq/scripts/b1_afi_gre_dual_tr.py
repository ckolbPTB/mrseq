"""Actual Flip Angle (AFI) imaging sequence - GRE with dual repetition times for B1 mapping."""

from pathlib import Path

import numpy as np
import pypulseq as pp

from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils import write_sequence


def b1_afi_gre_dual_tr_kernel(
    system: pp.Opts,
    te: float | None,
    tr1: float,
    tr2: float,
    fov_xy: float,
    n_readout: int,
    n_phase_encoding: int,
    n_dummy_excitations: int,
    slice_thickness: float,
    gx_pre_duration: float,
    gx_flat_time: float,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
    ge_segment_delay: float,
) -> tuple[pp.Sequence, float]:
    """Generate an AFI (Actual Flip Angle) sequence for B1 mapping.

    Acquires two complete FLASH images with different TR values to estimate
    the actual flip angle map via B1 inhomogeneity mapping.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr1
        First repetition time (TR1) (in seconds).
    tr2
        Second repetition time (TR2) (in seconds). Typically 5x TR1.
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_phase_encoding
        Number of phase encoding steps.
    n_dummy_excitations
        Number of dummy excitations before data acquisition to ensure steady state.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    gx_pre_duration
        Duration of readout pre-winder gradient (in seconds).
    gx_flat_time
        Flat time of readout gradient (in seconds).
    rf_duration
        Duration of the RF excitation pulse (in seconds).
    rf_flip_angle
        Flip angle of RF excitation pulse (in degrees).
    rf_bwt
        Bandwidth-time product of RF excitation pulse (Hz * seconds).
    rf_apodization
        Apodization factor of RF excitation pulse.
    ge_segment_delay
        Additional delay time after each readout for GE scanners (in seconds).

    Returns
    -------
    seq
        PyPulseq Sequence object.
    min_te
        Shortest possible echo time.

    """
    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # create slice selective excitation pulse and gradients
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=rf_flip_angle / 180 * np.pi,
        duration=rf_duration,
        slice_thickness=slice_thickness,
        apodization=rf_apodization,
        time_bw_product=rf_bwt,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
        use='excitation',
    )

    # create readout gradient and ADC
    delta_k = 1 / fov_xy
    gx = pp.make_trapezoid(channel='x', flat_area=n_readout * delta_k, flat_time=gx_flat_time, system=system)
    adc = pp.make_adc(num_samples=n_readout, duration=gx.flat_time, delay=gx.rise_time, system=system)

    # create frequency encoding pre- and re-winder gradient
    gx_pre = pp.make_trapezoid(
        channel='x',
        area=-gx.area / 2 - delta_k / 2,
        duration=gx_pre_duration,
        system=system,
        max_slew=system.max_slew * 0.7,
    )

    # phase encoding gradient for max ky position, reduce slew rate because all three gradients are on at the same time
    gy_pre_max = pp.make_trapezoid(
        channel='y',
        area=1 / fov_xy * n_readout / 2,
        duration=gx_pre_duration,
        system=system,
        max_slew=system.max_slew * 0.7,
    )

    # calculate gradient areas for (linear) phase encoding direction
    k0_center_id = np.where((np.arange(n_readout) - n_readout / 2) * delta_k == 0)[0][0]

    # create spoiler gradients
    gx_spoil = []
    gx_spoil.append(
        pp.make_trapezoid(
            channel='x',
            area=120 / slice_thickness,
            system=system,
            max_slew=system.max_slew,
        )
    )
    gx_spoil.append(
        pp.make_trapezoid(
            channel='x',
            area=tr2 / tr1 * 120 / slice_thickness,
            system=system,
            max_slew=system.max_slew,
        )
    )

    # calculate minimum echo time
    min_te = (
        rf.shape_dur / 2  # time from center to end of RF pulse
        + max(rf.ringdown_time, gz.fall_time)  # RF ringdown or gradient fall time
        + pp.calc_duration(gzr)  # slice selection rewinder gradient
        + pp.calc_duration(gx_pre)  # readout pre-winder gradient
        + gx.delay  # potential delay of readout gradient
        + gx.rise_time  # rise time of readout gradient
        + (k0_center_id + 0.5) * adc.dwell  # time to k-space center sample
    ).item()

    # calculate delay to achieve desired echo time
    if te is None:
        te_delay = 0.0
    else:
        te_delay = round_to_raster(te - min_te, system.block_duration_raster)
        if te_delay < 0:
            raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')

    print(f'\nMinimum TE: {min_te * 1000:.3f} ms')

    # rf spoiling
    rf_spoiling_phase_increment = 117
    rf_phase = 0.0
    rf_inc = 0.0

    for pe in range(-n_dummy_excitations, n_phase_encoding):
        # phase encoding along se and pe
        if pe >= 0:
            gy_pre = pp.scale_grad(gy_pre_max, (pe - n_phase_encoding / 2) / (n_phase_encoding / 2))
            pe_label = pp.make_label(type='SET', label='LIN', value=int(pe))
        else:
            gy_pre = pp.scale_grad(gy_pre_max, 0)
            pe_label = pp.make_label(type='SET', label='LIN', value=0)

        # loop over TR variants (TR1 and TR2)
        for tr_idx, tr_current in enumerate([tr1, tr2]):
            # set contrast ('ECO') label for current TR
            contrast_label = pp.make_label(type='SET', label='ECO', value=int(tr_idx))

            # save start time of current TR block
            _start_time_tr_block = sum(seq.block_durations.values())

            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi

            # add slice selective excitation pulse
            seq.add_block(rf, gz, pp.make_label(type='SET', label='TRID', value=88 if pe < 0 else 1))
            seq.add_block(gzr)

            # update rf phase offset for the next excitation pulse
            rf_inc = divmod(rf_inc + rf_spoiling_phase_increment, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            # add echo time delay
            seq.add_block(pp.make_delay(te_delay))

            # add pre-winder gradients and labels
            seq.add_block(gx_pre, gy_pre, pe_label, contrast_label)

            # add readout gradient and ADC
            if pe >= 0:
                seq.add_block(gx, adc, pe_label)
            else:
                seq.add_block(gx, pp.make_delay(pp.calc_duration(adc)))

            # y re-winder and spoiler gradients
            seq.add_block(pp.scale_grad(gy_pre, -1))
            seq.add_block(gx_spoil[tr_idx])

            # calculate TR delay
            duration_tr_block = sum(seq.block_durations.values()) - _start_time_tr_block
            tr_delay = round_to_raster(tr_current - duration_tr_block - ge_segment_delay, system.block_duration_raster)

            if tr_delay < 0:
                raise ValueError(
                    f'TR must be larger than {duration_tr_block * 1000:.3f} ms. '
                    f'Current value is {tr_current * 1000:.3f} ms.'
                )

            seq.add_block(pp.make_delay(tr_delay))

    return seq, min_te


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr1: float = 25e-3,
    tr2: float = 250e-3,
    rf_flip_angle: float = 60,
    fov_xy: float = 256e-3,
    n_readout: int = 128,
    n_phase_encoding: int = 128,
    n_dummy_excitations: int = 20,
    slice_thickness: float = 5e-3,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
    v141_compatibility: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate an AFI (Actual Flip Angle) sequence for B1 mapping.

    Acquires two complete FLASH images with different TR values (TR1 and TR2)
    to estimate the actual flip angle map. The B1 map can be reconstructed
    from the ratio of the two images.

    Parameters
    ----------
    system
        PyPulseq system limits object. Uses defaults if None.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if None.
    tr1
        First repetition time (TR1) (in seconds).
    tr2
        Second repetition time (TR2) (in seconds).
    rf_flip_angle
        Flip angle of rf excitation pulse (in degrees)
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_phase_encoding
        Number of phase encoding steps.
    n_dummy_excitations
        Number of dummy excitations before data acquisition to ensure steady state.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    show_plots
        Toggles sequence plot visualization.
    test_report
        Toggles advanced test report.
    timing_check
        Toggles timing check of the sequence.
    v141_compatibility
        Save the sequence in pulseq v1.4.1 for backwards compatibility.

    Returns
    -------
    seq
        Sequence object of AFI GRE B1 mapping sequence.
    file_path
        Path to the sequence file.

    """
    if system is None:
        system = sys_defaults

    # validate TR ratio
    tr_ratio = tr2 / tr1
    if tr_ratio < 3 or tr_ratio > 10:
        print(f'\nWarning: TR2/TR1 ratio is {tr_ratio:.2f}. Recommended range is 3-10 for optimal B1 sensitivity.')

    # define ADC and gradient timing
    adc_dwell = system.grad_raster_time
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time = n_readout * adc_dwell  # flat time of readout gradient [s]

    # define settings of RF excitation pulse
    rf_duration = 1.28e-3  # duration of the RF excitation pulse [s]
    rf_bwt = 4  # bandwidth-time product of RF excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of RF excitation pulse

    seq, min_te = b1_afi_gre_dual_tr_kernel(
        system=system,
        te=te,
        tr1=tr1,
        tr2=tr2,
        fov_xy=fov_xy,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        n_dummy_excitations=n_dummy_excitations,
        slice_thickness=slice_thickness,
        gx_pre_duration=gx_pre_duration,
        gx_flat_time=gx_flat_time,
        rf_duration=rf_duration,
        rf_flip_angle=rf_flip_angle,
        rf_bwt=rf_bwt,
        rf_apodization=rf_apodization,
        ge_segment_delay=0.0,
    )

    # check timing of the sequence
    if timing_check and not test_report:
        ok, error_report = seq.check_timing()
        if ok:
            print('\nTiming check passed successfully')
        else:
            print('\nTiming check failed! Error listing follows\n')
            print(error_report)

    # show advanced test report
    if test_report:
        print('\nCreating advanced test report...')
        print(seq.test_report())

    # define sequence filename
    filename = (
        f'{Path(__file__).stem}_'
        f'{int(fov_xy * 1000)}fov_'
        f'{n_readout}nx_{n_phase_encoding}ny_'
        f'TR1{int(tr1 * 1e3)}_TR2{int(tr2 * 1e3)}'
    )

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_xy, fov_xy, slice_thickness])
    seq.set_definition('ReconMatrix', (n_readout, n_phase_encoding, 1))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('TE', te or min_te)
    seq.set_definition('TR1', tr1)
    seq.set_definition('TR2', tr2)
    seq.set_definition('TRratio', tr2 / tr1)
    seq.set_definition('FlipAngle', rf_flip_angle)

    # save seq-file to disk
    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    write_sequence(seq, str(output_path / filename), create_signature=True, v141_compatibility=v141_compatibility)

    if show_plots:
        seq.plot(time_range=(0, tr1 + tr2))

    return seq, output_path / filename


if __name__ == '__main__':
    main()
