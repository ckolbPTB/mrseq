"""3D Cartesian FLASH sequence."""

from pathlib import Path

import numpy as np
import pypulseq as pp

from mrseq.utils import find_gx_flat_time_on_adc_raster
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils import write_sequence


def cartesian_flash_kernel(
    system: pp.Opts,
    te: float | None,
    tr: float | None,
    fov_xy: float,
    fov_z: float,
    n_readout: int,
    readout_oversampling: int,
    n_phase_encoding: int,
    n_slice_encoding: int,
    n_dummy_excitations: int,
    gx_pre_duration: float,
    gx_flat_time: float,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
    rf_spoiling_phase_increment: float,
    gz_spoil_duration: float,
    gz_spoil_area: float,
    ge_segment_delay: float,
) -> tuple[pp.Sequence, float, float]:
    """Generate a 3D Cartesian FLASH sequence.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    fov_z
        Field of view in the z direction (slice thickness) in meters.
    n_readout
        Number of frequency encoding steps.
    readout_oversampling
        Readout oversampling factor, commonly 2. This reduces aliasing artifacts.
    n_phase_encoding
        Number of phase encoding steps.
    n_slice_encoding
        Number of slice encoding steps.
    n_dummy_excitations
        Number of dummy excitations before data acquisition to ensure steady state.
    gx_pre_duration
        Duration of readout pre-winder gradient (in seconds)
    gx_flat_time
        Flat time of readout gradient (in seconds)
    rf_duration
        Duration of the rf excitation pulse (in seconds)
    rf_flip_angle
        Flip angle of rf excitation pulse (in degrees)
    rf_bwt
        Bandwidth-time product of rf excitation pulse (Hz * seconds)
    rf_apodization
        Apodization factor of rf excitation pulse
    rf_spoiling_phase_increment
        RF spoiling phase increment (in degrees). Set to 0 for no RF spoiling.
    gz_spoil_duration
        Duration of spoiler gradient (in seconds)
    gz_spoil_area
        Area of spoiler gradient (in mT/m * s)
    ge_segment_delay
        Delay time at the end of each segment for GE scanners.

    Returns
    -------
    seq
        PyPulseq Sequence object
    min_te
        Shortest possible echo time.
    min_tr
        Shortest possible repetition time.

    """
    if readout_oversampling < 1:
        raise ValueError('Readout oversampling factor must be >= 1.')

    if n_dummy_excitations < 0:
        raise ValueError('Number of dummy excitations must be >= 0.')

    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # create slice selective excitation pulse and gradients
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
    delta_k = 1 / fov_xy
    gx = pp.make_trapezoid(channel='x', flat_area=n_readout * delta_k, flat_time=gx_flat_time, system=system)
    n_readout_with_oversampling = int(n_readout * readout_oversampling)
    n_readout_with_oversampling = n_readout_with_oversampling + np.mod(n_readout_with_oversampling, 2)  # make even
    adc = pp.make_adc(num_samples=n_readout_with_oversampling, duration=gx.flat_time, delay=gx.rise_time, system=system)

    # create readout pre- and re-winder gradient, reduce slew rate because all three gradients are on at the same time
    gx_pre = pp.make_trapezoid(
        channel='x',
        area=-gx.area / 2 - delta_k / 2,
        duration=gx_pre_duration,
        system=system,
        max_slew=system.max_slew * 0.7,
    )
    gx_post = pp.make_trapezoid(
        channel='x',
        area=-gx.area / 2 + delta_k / 2,
        duration=gx_pre_duration,
        system=system,
        max_slew=system.max_slew * 0.7,
    )
    k0_center_id = np.where((np.arange(n_readout_with_oversampling) - n_readout_with_oversampling / 2) * delta_k == 0)[
        0
    ][0]

    # phase encoding gradient, reduce slew rate because all three gradients are on at the same time
    gy_pre_max = pp.make_trapezoid(
        channel='y',
        area=delta_k * n_phase_encoding / 2,
        duration=gx_pre_duration,
        system=system,
        max_slew=system.max_slew * 0.7,
    )

    # slice encoding gradient, reduce slew rate because all three gradients are on at the same time
    gz_pre_max = pp.make_trapezoid(
        channel='z',
        area=1 / fov_z * n_slice_encoding / 2,
        duration=gx_pre_duration,
        system=system,
        max_slew=system.max_slew * 0.7,
    )

    # create spoiler gradients, reduce slew rate because all three gradients are on at the same time
    gz_spoil = pp.make_trapezoid(
        channel='z', system=system, area=gz_spoil_area, duration=gz_spoil_duration, max_slew=system.max_slew * 0.7
    )

    # calculate minimum echo time
    gzr_gx_dur = pp.calc_duration(gzr) + pp.calc_duration(gx_pre)  # gzr and gx_pre are applied sequentially

    min_te = (
        rf.shape_dur / 2  # time from center to end of RF pulse
        + max(rf.ringdown_time, gz.fall_time)  # RF ringdown time or gradient fall time
        + gzr_gx_dur  # slice selection re-phasing gradient and readout pre-winder
        + gx.delay  # potential delay of readout gradient
        + gx.rise_time  # rise time of readout gradient
        + (k0_center_id + 0.5) * adc.dwell  # time from beginning of ADC to time point of k-space center sample
    ).item()

    # calculate echo time delay (te_delay)
    if te is None:
        te_delay = 0.0
    else:
        te_delay = round_to_raster(te - min_te, system.block_duration_raster)
        if te_delay < 0:
            raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')

    # calculate minimum repetition time
    min_tr = (
        pp.calc_duration(gz)  # rf pulse
        + gzr_gx_dur  # slice selection re-phasing gradient and readout pre-winder
        + pp.calc_duration(gx)  # readout gradient
        + pp.calc_duration(gz_spoil, gx_post)  # gradient spoiler or readout-re-winder
    )

    # calculate repetition time delay (tr_delay)
    current_min_tr = min_tr + te_delay + ge_segment_delay
    if tr is None:
        tr_delay = 0.0
    else:
        tr_delay = round_to_raster(tr - current_min_tr, system.block_duration_raster)
        if not tr_delay >= 0:
            raise ValueError(
                f'TR must be larger than {current_min_tr * 1000:.3f} ms. Current value is {tr * 1000:.3f} ms.'
            )

    print(f'\nCurrent echo time = {(min_te + te_delay) * 1000:.3f} ms')
    print(f'Current repetition time = {(current_min_tr + tr_delay) * 1000:.3f} ms')

    # choose initial rf phase offset
    rf_phase = 0.0
    rf_inc = 0.0

    for se in range(n_slice_encoding):
        n_dummy = n_dummy_excitations if se == 0 else 0

        for pe in range(-n_dummy, n_phase_encoding):
            # phase encoding along se and pe
            if pe >= 0:
                gz_pre = pp.scale_grad(gz_pre_max, (se - n_slice_encoding / 2) / (n_slice_encoding / 2))
                se_label = pp.make_label(type='SET', label='PAR', value=int(se))
                gy_pre = pp.scale_grad(gy_pre_max, (pe - n_phase_encoding / 2) / (n_phase_encoding / 2))
                pe_label = pp.make_label(type='SET', label='LIN', value=int(pe))
            else:
                gz_pre = pp.scale_grad(gz_pre_max, 0)
                se_label = pp.make_label(type='SET', label='PAR', value=0)
                gy_pre = pp.scale_grad(gy_pre_max, 0)
                pe_label = pp.make_label(type='SET', label='LIN', value=0)

            # calculate current phase_offset if rf_spoiling is activated
            if rf_spoiling_phase_increment > 0:
                rf.phase_offset = rf_phase / 180 * np.pi
                adc.phase_offset = rf_phase / 180 * np.pi

            # add slice selective excitation pulse
            seq.add_block(rf, gz, pp.make_label(type='SET', label='TRID', value=88 if pe < 0 else 1))

            # update rf phase offset for the next excitation pulse
            rf_inc = divmod(rf_inc + rf_spoiling_phase_increment, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            seq.add_block(gzr)
            seq.add_block(pp.make_delay(te_delay))
            seq.add_block(gx_pre, gy_pre, gz_pre)

            if pe >= 0:
                seq.add_block(gx, adc, pe_label, se_label)
            else:
                seq.add_block(gx, pp.make_delay(pp.calc_duration(adc)))

            seq.add_block(gx_post, pp.scale_grad(gy_pre, -1), gz_spoil)

            # add delay in case TR > min_TR
            if tr_delay > 0:
                seq.add_block(pp.make_delay(tr_delay - ge_segment_delay))

    # obtain noise samples
    seq.add_block(
        pp.make_delay(0.1),
        pp.make_label(label='LIN', type='SET', value=0),
        pp.make_label(label='SLC', type='SET', value=0),
        pp.make_label(type='SET', label='TRID', value=99),
    )
    seq.add_block(adc, pp.make_label(label='NOISE', type='SET', value=True))
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    return seq, min_te, min_tr


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr: float | None = None,
    rf_flip_angle: float = 12,
    fov_xy: float = 128e-3,
    fov_z: float = 80e-3,
    n_readout: int = 128,
    n_phase_encoding: int = 128,
    n_slice_encoding=10,
    receiver_bandwidth_per_pixel: float = 800,  # Hz/pixel
    n_dummy_excitations: int = 20,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
    v141_compatibility: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate a 3D Cartesian FLASH sequence.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds). Minimum repetition time is used if set to None.
    rf_flip_angle
        Flip angle of rf excitation pulse (in degrees)
    fov_xy
        Field of view in x and y direction (in meters).
    fov_z
        Field of view along z (in meters).
    n_readout
        Number of frequency encoding steps.
    n_phase_encoding
        Number of phase encoding steps.
    n_slice_encoding
        Number of phase encoding steps along the slice direction.
    receiver_bandwidth_per_pixel
        Desired receiver bandwidth per pixel (in Hz/pixel). This is used to calculate the readout duration.
    n_dummy_excitations
        Number of dummy excitations before data acquisition to ensure steady state.
    show_plots
        Toggles sequence plot.
    test_report
        Toggles advanced test report.
    timing_check
        Toggles timing check of the sequence.
    v141_compatibility
        Save the sequence in pulseq v1.4.1 for backwards compatibility.

    Returns
    -------
    seq
        Sequence object of 3D Cartesian FLASH sequence.
    file_path
        Path to the sequence file.
    """
    if system is None:
        system = sys_defaults

    # define settings of rf excitation pulse
    rf_duration = 1.28e-3  # duration of the rf excitation pulse [s]
    rf_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse
    readout_oversampling = 2  # readout oversampling factor, commonly 2. This reduces aliasing artifacts.

    # define ADC and gradient timing
    n_readout_with_oversampling = int(n_readout * readout_oversampling)
    adc_dwell_time = 1.0 / (receiver_bandwidth_per_pixel * n_readout_with_oversampling)
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time, adc_dwell_time = find_gx_flat_time_on_adc_raster(
        n_readout_with_oversampling, adc_dwell_time, system.grad_raster_time, system.adc_raster_time
    )

    # define spoiling
    gz_spoil_duration = 0.8e-3  # duration of spoiler gradient [s]
    gz_spoil_area = 4 / (fov_z / n_slice_encoding)  # area / zeroth gradient moment of spoiler gradient
    rf_spoiling_phase_increment = 117  # RF spoiling phase increment [°]. Set to 0 for no RF spoiling.

    seq, min_te, min_tr = cartesian_flash_kernel(
        system=system,
        te=te,
        tr=tr,
        fov_xy=fov_xy,
        fov_z=fov_z,
        n_readout=n_readout,
        readout_oversampling=readout_oversampling,
        n_phase_encoding=n_phase_encoding,
        n_slice_encoding=n_slice_encoding,
        n_dummy_excitations=n_dummy_excitations,
        gx_pre_duration=gx_pre_duration,
        gx_flat_time=gx_flat_time,
        rf_duration=rf_duration,
        rf_flip_angle=rf_flip_angle,
        rf_bwt=rf_bwt,
        rf_apodization=rf_apodization,
        rf_spoiling_phase_increment=rf_spoiling_phase_increment,
        gz_spoil_duration=gz_spoil_duration,
        gz_spoil_area=gz_spoil_area,
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

    # show advanced rest report
    if test_report:
        print('\nCreating advanced test report...')
        print(seq.test_report())

    filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_xy_{int(fov_z * 1000)}_fov_z_'
    filename += f'{n_readout}nx_{n_phase_encoding}ny_{n_slice_encoding}nz'

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_xy, fov_xy, fov_z])
    seq.set_definition('ReconMatrix', (n_readout, n_phase_encoding, n_slice_encoding))
    seq.set_definition('TE', te or min_te)
    seq.set_definition('TR', tr or min_tr)
    seq.set_definition('ReadoutOversamplingFactor', readout_oversampling)

    # save seq-file to disk
    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    write_sequence(seq, str(output_path / filename), create_signature=True, v141_compatibility=v141_compatibility)

    if show_plots:
        seq.plot(time_range=(0, 10 * (tr or min_tr)))

    return seq, output_path / filename


if __name__ == '__main__':
    main()
