"""Actual Flip Angle (AFI) imaging sequence - GRE with dual repetition times for B1 mapping."""

from pathlib import Path

import numpy as np
import pypulseq as pp

from mrseq.preparations.receiver_gain_calibration import add_gre_receiver_gain_calibration
from mrseq.utils import find_gx_flat_time_on_adc_raster
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
    fov_z: float,
    n_slice_encoding: int,
    gx_pre_duration: float,
    gx_flat_time: float,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
    gx_spoil_area: float,
    gx_spoil_slew_rate: float,
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
    fov_z
        Field of view in z direction (in meters).
    n_slice_encoding
        Number of slice encoding steps.
        For n_slice_encoding  = 1 a slice selective pulse is used, otherwise a block pulse is used for excitation.
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
    gx_spoil_area
        Area of spoiler gradient (in mT/m * s)
    gx_spoil_slew_rate
        Max slew rate of spoiler gradient
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
    if n_slice_encoding == 1:
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
    else:
        rf = pp.make_block_pulse(
            flip_angle=rf_flip_angle * np.pi / 180,
            delay=system.rf_dead_time,
            duration=rf_duration,
            system=system,
            use='excitation',
        )

    # create readout gradient and ADC
    delta_k = 1 / fov_xy
    gx = pp.make_trapezoid(channel='x', flat_area=n_readout * delta_k, flat_time=gx_flat_time, system=system)
    adc = pp.make_adc(num_samples=n_readout, duration=gx.flat_time, delay=gx.rise_time, system=system)

    print(
        f'Receiver bandwidth: {int(1.0 / (adc.num_samples * adc.dwell))} Hz/pixel '
        f'(Readout duration: {adc.num_samples * adc.dwell * 1000:.3f} ms).'
    )

    # create frequency encoding pre- and re-winder gradient
    gx_pre = pp.make_trapezoid(
        channel='x',
        area=-gx.area / 2 - delta_k / 2,
        duration=gx_pre_duration,
        system=system,
    )

    # phase encoding gradient for max ky position
    gy_pre_max = pp.make_trapezoid(
        channel='y',
        area=1 / fov_xy * n_phase_encoding / 2,
        duration=gx_pre_duration,
        system=system,
    )

    # slice encoding gradient
    gz_pre_max = pp.make_trapezoid(
        channel='z',
        area=1 / fov_z * n_slice_encoding / 2,
        duration=gx_pre_duration,
        system=system,
    )

    # calculate gradient areas for (linear) phase encoding direction
    k0_center_id = np.where((np.arange(n_readout) - n_readout / 2) * delta_k == 0)[0][0]

    # create spoiler gradients
    gx_spoil = []
    gx_spoil.append(
        pp.make_trapezoid(
            channel='x',
            area=gx_spoil_area,
            system=system,
            max_slew=gx_spoil_slew_rate,
        )
    )
    gx_spoil.append(
        pp.make_trapezoid(
            channel='x',
            area=tr2 / tr1 * gx_spoil_area,
            system=system,
            max_slew=gx_spoil_slew_rate,
        )
    )

    # calculate minimum echo time
    gz_time = max(rf.ringdown_time, gz.fall_time) + pp.calc_duration(gzr) if n_slice_encoding == 1 else rf.ringdown_time
    min_te = (
        rf.shape_dur / 2  # time from center to end of RF pulse
        + gz_time
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

    if ge_segment_delay > 0:
        seq, _ = add_gre_receiver_gain_calibration(
            system=system,
            seq=seq,
            rf_flip_angle=rf_flip_angle,
            te=te_delay + min_te,
            fov_z=fov_z,
        )
        seq.add_block(pp.make_delay(1.0))

    # rf spoiling
    rf_spoiling_phase_increment = 117
    rf_phase = 0.0
    rf_inc = 0.0

    for se in range(n_slice_encoding):
        for pe in range(-n_dummy_excitations if se == 0 else 0, n_phase_encoding):
            # phase encoding along se and pe
            if pe >= 0:
                gz_pre = pp.scale_grad(gz_pre_max, (se - n_slice_encoding // 2) / (n_slice_encoding / 2))
                se_label = pp.make_label(type='SET', label='PAR', value=int(se))
                gy_pre = pp.scale_grad(gy_pre_max, (pe - n_phase_encoding // 2) / (n_phase_encoding / 2))
                pe_label = pp.make_label(type='SET', label='LIN', value=int(pe))
            else:
                gz_pre = pp.scale_grad(gz_pre_max, 0)
                se_label = pp.make_label(type='SET', label='PAR', value=0)
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
                trid_label = 100 + int(tr_idx + 1) if pe < 0 else int(tr_idx + 1)
                if n_slice_encoding == 1:
                    seq.add_block(rf, gz, pp.make_label(type='SET', label='TRID', value=trid_label))
                    seq.add_block(gzr)
                else:
                    seq.add_block(rf, pp.make_label(type='SET', label='TRID', value=trid_label))

                # update rf phase offset for the next excitation pulse
                rf_inc = divmod(rf_inc + rf_spoiling_phase_increment, 360.0)[1]
                rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

                # add echo time delay
                seq.add_block(pp.make_delay(te_delay))

                # add pre-winder gradients and labels
                seq.add_block(gx_pre, gy_pre, gz_pre, pe_label, contrast_label, se_label)

                # add readout gradient and ADC
                if pe >= 0:
                    seq.add_block(gx, adc, pe_label)
                else:
                    seq.add_block(gx, pp.make_delay(pp.calc_duration(adc)))

                # y re-winder and spoiler gradients
                seq.add_block(pp.scale_grad(gy_pre, -1), pp.scale_grad(gz_pre, -1))
                seq.add_block(gx_spoil[tr_idx])

                # calculate TR delay
                duration_tr_block = sum(seq.block_durations.values()) - _start_time_tr_block
                tr_delay = round_to_raster(
                    tr_current - duration_tr_block - ge_segment_delay, system.block_duration_raster
                )

                if tr_delay < 0:
                    raise ValueError(
                        f'TR must be larger than {duration_tr_block * 1000:.3f} ms. '
                        f'Current value is {tr_current * 1000:.3f} ms.'
                    )

                seq.add_block(pp.make_delay(tr_delay))

    # obtain noise samples
    seq.add_block(
        pp.make_delay(0.1),
        pp.make_label(label='LIN', type='SET', value=0),
        pp.make_label(label='SLC', type='SET', value=0),
        pp.make_label(type='SET', label='TRID', value=9999),
    )
    seq.add_block(
        adc,
        pp.make_delay(round_to_raster(pp.calc_duration(adc), system.block_duration_raster, 'ceil')),
        pp.make_label(label='NOISE', type='SET', value=True),
    )
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

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
    fov_z: float = 8e-3,
    n_slice_encoding: int = 1,
    receiver_bandwidth_per_pixel: float = 600,  # Hz/pixel
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
    fov_z
        Field of view in z direction (in meters).
    n_slice_encoding
        Number of slice encoding steps.
        For n_slice_encoding  = 1 a slice selective pulse is used, otherwise a block pulse is used for excitation.
    receiver_bandwidth_per_pixel
        Desired receiver bandwidth per pixel (in Hz/pixel). This is used to calculate the readout duration.
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
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]
    adc_dwell_time = round_to_raster(1.0 / (receiver_bandwidth_per_pixel * n_readout), system.adc_raster_time)
    gx_flat_time, adc_dwell_time = find_gx_flat_time_on_adc_raster(
        n_readout, adc_dwell_time, system.grad_raster_time, system.adc_raster_time
    )

    # define settings of RF excitation pulse
    rf_duration = 1.28e-3  # duration of the RF excitation pulse [s]
    rf_bwt = 4  # bandwidth-time product of RF excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of RF excitation pulse

    # gradient moment for spoiling of first TR: 300 mT*ms/m
    gx_spoil_area = 300 * 1e-6 * system.gamma
    gx_spoil_slew_rate = min(system.max_slew, 70 * system.gamma)

    seq, min_te = b1_afi_gre_dual_tr_kernel(
        system=system,
        te=te,
        tr1=tr1,
        tr2=tr2,
        fov_xy=fov_xy,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        n_dummy_excitations=n_dummy_excitations,
        fov_z=fov_z,
        n_slice_encoding=n_slice_encoding,
        gx_pre_duration=gx_pre_duration,
        gx_flat_time=gx_flat_time,
        rf_duration=rf_duration,
        rf_flip_angle=rf_flip_angle,
        rf_bwt=rf_bwt,
        rf_apodization=rf_apodization,
        gx_spoil_area=gx_spoil_area,
        gx_spoil_slew_rate=gx_spoil_slew_rate,
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
    seq.set_definition('FOV', [fov_xy, fov_xy, fov_z])
    seq.set_definition('ReconMatrix', (n_readout, n_phase_encoding, n_slice_encoding))
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
