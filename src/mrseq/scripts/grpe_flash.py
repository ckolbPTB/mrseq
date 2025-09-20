"""3D FLASH sequence with radial phase encoding."""

from pathlib import Path

import ismrmrd
import numpy as np
import pypulseq as pp
from pypulseq.rotate import rotate

from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils.create_ismrmrd_header import create_header


def grpe_flash_kernel(
    system: pp.Opts,
    te: float | None,
    tr: float | None,
    fov_x: float,
    fov_y: float,
    fov_z: float,
    n_readout: int,
    n_rpe_points: int,
    n_rpe_points_per_shot: int,
    n_rpe_spokes: int,
    readout_oversampling: float,
    partial_fourier_factor: float,
    n_dummy_excitations: int,
    gx_pre_duration: float,
    gx_flat_time: float,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
    rf_spoiling_phase_increment: float,
    gx_spoil_duration: float,
    gx_spoil_area: float,
    mrd_header_file: str | None,
) -> tuple[pp.Sequence, float, float]:
    """Generate a 3D FLASH sequence with golden radial phase encoding.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds).
    fov_x
        Field of view in x direction (in meters).
    fov_y
        Field of view in y direction (in meters).
    fov_z
        Field of view in z direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_rpe_points
        Number of radial phase encoding points (points along one RPE line).
    n_rpe_points_per_shot
        Shots are interleaved groups of points along RPE lines. Each shot obtains the k-space center at the cost of a
        point of the highest k-space frequency. Fat-saturation pulses are applied prior to each shot.
    n_rpe_spokes
        Number of radial phase encoding spokes (number of RPE lines).
    readout_oversampling
        Readout oversampling factor, commonly 2. This reduces aliasing artifacts.
    partial_fourier_factor
        Partial Fourier factor along RPE lines (between 0.5 and 1).
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
    gx_spoil_duration
        Duration of spoiler gradient (in seconds)
    gx_spoil_area
        Area of spoiler gradient (in mT/m * s)
    mrd_header_file
        Filename of the ISMRMRD header file to be created. If None, no header file is created.

    Returns
    -------
    seq
        PyPulseq Sequence object
    min_te
        Shortest possible echo time.

    """
    spoke_angle = np.pi * 0.618034
    rpe_radial_shift = [0, 0.5, 0.25, 0.75]

    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # create slab selective excitation pulse and gradients
    rf, gz, gzr = pp.make_sinc_pulse(  # type: ignore
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
    delta_k = 1 / fov_x
    gx = pp.make_trapezoid(channel='x', flat_area=n_readout * delta_k, flat_time=gx_flat_time, system=system)
    n_readout_with_oversampling = int(n_readout * readout_oversampling)
    n_readout_with_oversampling = n_readout_with_oversampling + np.mod(n_readout_with_oversampling, 2)  # make even
    adc = pp.make_adc(num_samples=n_readout_with_oversampling, duration=gx.flat_time, delay=gx.rise_time, system=system)

    # create frequency encoding pre- and re-winder gradient
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2 - delta_k / 2, duration=gx_pre_duration, system=system)
    k0_center_id = np.where((np.arange(n_readout) - n_readout / 2) * delta_k == 0)[0][0]

    # calculate gradient areas for phase encoding along each RPE line in a low-high order
    delta_ky = 1 / fov_y
    n_rpe_points = int(((n_rpe_points * partial_fourier_factor) // n_rpe_points_per_shot) * n_rpe_points_per_shot)
    print(f'Number of phase encoding points {n_rpe_points} with partial Fourier factor {partial_fourier_factor}')
    enc_steps_pe = np.arange(0, n_rpe_points)
    phase_areas = (enc_steps_pe - n_rpe_points / 2) * delta_ky
    centric_idx = np.argsort(np.abs(phase_areas), kind='stable')
    enc_steps_pe = enc_steps_pe[centric_idx]

    # Interleave shots
    enc_steps_pe_interleaved = []
    for step in range(n_rpe_points // n_rpe_points_per_shot):
        enc_steps_pe_interleaved.append(enc_steps_pe[step :: n_rpe_points // n_rpe_points_per_shot])
    enc_steps_pe = np.concatenate(enc_steps_pe_interleaved)

    enc_steps_se = np.arange(0, n_rpe_spokes)

    # create spoiler gradients
    gx_spoil = pp.make_trapezoid(channel='x', system=system, area=gx_spoil_area, duration=gx_spoil_duration)

    # calculate minimum echo time
    if te is None:
        gzr_gx_dur = pp.calc_duration(gzr, gx_pre)  # gzr and gx_pre are applied simultaneously
    else:
        gzr_gx_dur = pp.calc_duration(gzr) + pp.calc_duration(gx_pre)  # gzr and gx_pre are applied sequentially

    min_te = (
        pp.calc_duration(gz) / 2  # half duration of rf pulse
        + gzr_gx_dur  # slice selection re-phasing gradient and readout pre-winder
        + gx.delay  # potential delay of readout gradient
        + gx.rise_time  # rise time of readout gradient
        + (k0_center_id + 0.5) * adc.dwell  # time from beginning of ADC to time point of k-space center sample
    )

    # calculate echo time delay (te_delay)
    te_delay = 0 if te is None else round_to_raster(te - min_te, system.block_duration_raster)
    if not te_delay >= 0:
        raise ValueError(f'TE must be larger than {min_te * 1000:.2f} ms. Current value is {te * 1000:.2f} ms.')

    # calculate minimum repetition time
    min_tr = (
        pp.calc_duration(gz)  # rf pulse
        + gzr_gx_dur  # slice selection re-phasing gradient and readout pre-winder
        + pp.calc_duration(gx)  # readout gradient
        + pp.calc_duration(gx_spoil)  # gradient spoiler
    )

    # calculate repetition time delay (tr_delay)
    current_min_tr = min_tr + te_delay
    tr_delay = 0 if tr is None else round_to_raster(tr - current_min_tr, system.block_duration_raster)

    if not tr_delay >= 0:
        raise ValueError(f'TR must be larger than {current_min_tr * 1000:.2f} ms. Current value is {tr * 1000:.2f} ms.')

    print(f'\nCurrent echo time = {(min_te + te_delay) * 1000:.2f} ms')
    print(f'Current repetition time = {(current_min_tr + tr_delay) * 1000:.2f} ms')

    # choose initial rf phase offset
    rf_phase = 0
    rf_inc = 0

    # create header
    if mrd_header_file:
        hdr = create_header(
            traj_type='other',
            fov=fov_x,
            res=fov_x / n_readout,
            slice_thickness=fov_z,
            dt=adc.dwell,
            n_k1=n_rpe_points,
            n_k2=n_rpe_spokes,
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    # obtain noise samples
    # seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='SLC', type='SET', value=0))
    # seq.add_block(adc, pp.make_label(label='NOISE', type='SET', value=True))
    # seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))

    # init labels (still required - missing SET label will prohibit the sequence from running)
    seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='SLC', type='SET', value=0))

    # Define sequence blocks
    for se in enc_steps_se:
        se_label = pp.make_label(type='SET', label='PAR', value=int(se))
        for pe_index in range(-n_dummy_excitations, n_rpe_points):
            # if use_fat_sat and np.mod(pe, n_rpe_points) == 0:
            #    # add fat-sat pulse and gradient
            #    seq.add_block(rf_fs, gz_fs)

            pe = enc_steps_pe[pe_index] if pe_index >= 0 else 0
            pe_label = pp.make_label(type='SET', label='LIN', value=int(pe))

            # set rf pulse properties and add rf pulse block event
            if rf_spoiling_phase_increment > 0:
                rf.phase_offset = rf_phase / 180 * np.pi
                adc.phase_offset = rf_phase / 180 * np.pi

            # add rf pulse
            seq.add_block(rf, gz)

            # update rf pulse properties for the next loop
            rf_inc = divmod(rf_inc + rf_spoiling_phase_increment, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            # area of phase encoding gradient
            current_phase_area = phase_areas[int(pe)]
            if np.abs(current_phase_area) > 1e-5:  # do not shift k-space center for respiratory navigator calculation
                current_phase_area += rpe_radial_shift[int(np.mod(se, len(rpe_radial_shift)))] * delta_ky

            # phase encoding along pe
            gy_pre = pp.make_trapezoid(
                channel='y',
                area=current_phase_area,
                duration=pp.calc_duration(gx_pre),
                system=system,
            )

            # calculate rotated phase encoding gradient
            rotation_angle_rad = spoke_angle * se
            gy_pre_rotated = rotate(gy_pre, angle=rotation_angle_rad, axis='x')
            gy_pre = gy_pre_rotated[0]
            if len(gy_pre_rotated) == 2:
                gz_pre = gy_pre_rotated[1]
            else:
                gz_pre = pp.make_trapezoid(
                    channel='z',
                    area=0,
                    duration=pp.calc_duration(gx_pre),
                    system=system,
                )
            assert gy_pre.channel == 'y' and gz_pre.channel == 'z'

            # combine slice rephaser and slice encoding gradient
            gz_r_pre = pp.make_trapezoid(
                channel='z',
                area=gz_pre.area + gzr.area,
                duration=pp.calc_duration(gx_pre),
                system=system,
            )

            label_contents = [pe_label, se_label]
            seq.add_block(gx_pre, gy_pre, gz_r_pre, *label_contents)

            # add delay due to TE
            if te_delay > 0:
                seq.add_block(pp.make_delay(te_delay))

            # add readout gradient and ADC
            if pe_index >= 0:
                seq.add_block(gx, adc)

            # add re-winder and spoiler gradients
            gy_pre.amplitude = -gy_pre.amplitude
            gz_pre.amplitude = -gz_pre.amplitude
            seq.add_block(gx_spoil, gy_pre, gz_pre)

            # add delay in case TR > min_TR
            if tr_delay > 0:
                seq.add_block(pp.make_delay(tr_delay))

            if mrd_header_file and pe_index >= 0:
                # add acquisitions to metadata
                k0_trajectory = np.linspace(
                    -n_readout_with_oversampling // 2,
                    (n_readout_with_oversampling // 2) - 1,
                    n_readout_with_oversampling,
                )
                grpe_trajectory = np.zeros((n_readout_with_oversampling, 3), dtype=np.float32)

                grpe_trajectory[:, 0] = k0_trajectory
                grpe_trajectory[:, 1] = pe * np.cos(rotation_angle_rad)
                grpe_trajectory[:, 2] = pe * np.sin(rotation_angle_rad)

                acq = ismrmrd.Acquisition()
                acq.resize(trajectory_dimensions=3, number_of_samples=adc.num_samples)
                acq.traj[:] = grpe_trajectory
                prot.append_acquisition(acq)

    # close ISMRMRD file
    if mrd_header_file:
        prot.close()

    return seq, min_te, min_tr


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr: float | None = None,
    fov_x: float = 128e-3,
    fov_y: float = 128e-3,
    fov_z: float = 128e-3,
    n_readout: int = 128,
    n_rpe_points: int = 128,
    n_rpe_points_per_shot: int = 8,
    n_rpe_spokes: int = 16,
    partial_fourier_factor: float = 0.7,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
) -> pp.Sequence:
    """Generate a 3D FLASH sequence with radial phase encoding.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds). Minimum repetition time is used if set to None.
    fov_x
        Field of view in x direction (in meters).
    fov_y
        Field of view in y direction (in meters).
    fov_z
        Field of view in z direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_rpe_points
        Number of radial phase encoding points (points along one RPE line).
    n_rpe_points_per_shot
        Shots are interleaved groups of points along RPE lines. Each shot obtains the k-space center at the cost of a
        point of the highest k-space frequency. Fat-saturation pulses are applied prior to each shot.
    n_rpe_spokes
        Number of radial phase encoding spokes (number of RPE lines).
    partial_fourier_factor
        Partial Fourier factor along RPE lines (between 0.5 and 1).
    show_plots
        Toggles sequence plot.
    test_report
        Toggles advanced test report.
    timing_check
        Toggles timing check of the sequence.
    """
    if system is None:
        system = sys_defaults

    # define ADC and gradient timing
    adc_dwell = system.grad_raster_time
    gx_pre_duration = 0.8e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time = n_readout * adc_dwell  # flat time of readout gradient [s]

    # define settings of rf excitation pulse
    rf_duration = 0.6e-3  # duration of the rf excitation pulse [s]
    rf_flip_angle = 12  # flip angle of rf excitation pulse [°]
    rf_bwt = 2  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse
    readout_oversampling = 2  # readout oversampling factor, commonly 2. This reduces aliasing artifacts.

    n_dummy_excitations = 40  # number of dummy excitations before data acquisition to ensure steady state

    # define spoiling
    gx_spoil_duration = 1.9e-3  # duration of spoiler gradient [s]
    gx_spoil_area = readout_oversampling * n_readout * 1 / fov_x  # area / zeroth gradient moment of spoiler gradient
    rf_spoiling_phase_increment = 117  # RF spoiling phase increment [°]. Set to 0 for no RF spoiling.

    # define sequence filename
    filename = f'{Path(__file__).stem}_fov{int(fov_x * 1000)}_{int(fov_y * 1000)}_{int(fov_z * 1000)}mm_'
    filename += f'{n_readout}_{n_rpe_points}_{n_rpe_spokes}_3d'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '.mrd')).exists():
        (output_path / Path(filename + '.mrd')).unlink()

    seq, min_te, min_tr = grpe_flash_kernel(
        system=system,
        te=te,
        tr=tr,
        fov_x=fov_x,
        fov_y=fov_y,
        fov_z=fov_z,
        n_readout=n_readout,
        n_rpe_points=n_rpe_points,
        n_rpe_points_per_shot=n_rpe_points_per_shot,
        n_rpe_spokes=n_rpe_spokes,
        readout_oversampling=readout_oversampling,
        partial_fourier_factor=partial_fourier_factor,
        n_dummy_excitations=n_dummy_excitations,
        gx_pre_duration=gx_pre_duration,
        gx_flat_time=gx_flat_time,
        rf_duration=rf_duration,
        rf_flip_angle=rf_flip_angle,
        rf_bwt=rf_bwt,
        rf_apodization=rf_apodization,
        rf_spoiling_phase_increment=rf_spoiling_phase_increment,
        gx_spoil_duration=gx_spoil_duration,
        gx_spoil_area=gx_spoil_area,
        mrd_header_file=output_path / Path(filename + '.mrd'),
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

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov_x, fov_y, fov_z])
    seq.set_definition('ReconMatrix', (n_readout, n_readout, 1))
    seq.set_definition('SliceThickness', fov_z)
    seq.set_definition('TE', te or min_te)
    seq.set_definition('TR', tr or min_tr)

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    if show_plots:
        seq.plot(time_range=(0, 10 * (tr or min_tr)))

    return seq


if __name__ == '__main__':
    main()
