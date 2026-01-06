"""2D Echo Planar Imaging (EPI) sequence."""

from math import floor
from pathlib import Path
from typing import Literal

import ismrmrd
import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp

from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils.ismrmrd import Fov
from mrseq.utils.ismrmrd import Limits
from mrseq.utils.ismrmrd import MatrixSize
from mrseq.utils.ismrmrd import create_header
from mrseq.utils.trajectory import EpiReadout


def epi2d_kernel(
    system: pp.Opts,
    te: float | None,
    tr: float | None,
    fov: float,
    n_readout: int,
    n_phase_encoding: int,
    bandwidth: float,
    slice_thickness: float,
    n_slices: int,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
    readout_type: Literal['symmetric', 'flyback'],
    echo_type: Literal['FID', 'SE'],
    oversampling: Literal[1, 2, 4],
    ramp_sampling: bool,
    partial_fourier_factor: float,
    pe_enable: bool,
    spoiling_enable: bool,
    add_noise_acq: bool,
    add_navigator_acq: bool,
    mrd_header_file: str | Path | None,
) -> tuple[pp.Sequence, float, float]:
    """Generate a 2D Echo Planar Imaging (EPI) sequence.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds).
    fov
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_phase_encoding
        Number of phase encoding steps.
    bandwidth
        Total receiver bandwidth in Hz.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    n_slices
        Number of slices.
    rf_duration
        Duration of the rf excitation pulse (in seconds)
    rf_flip_angle
        Flip angle of rf excitation pulse (in degrees)
    rf_bwt
        Bandwidth-time product of rf excitation pulse (Hz * seconds)
    rf_apodization
        Apodization factor of rf excitation pulse
    readout_type
        Readout type ('symmetric' or 'flyback').
    echo_type
        Echo type ('FID' or 'SE').
    oversampling
        ADC oversampling factor.
    ramp_sampling
        If True, ADC is active during gradient ramps (optimized timing).
    partial_fourier_factor
        Partial Fourier factor (0.5 to 1.0).
    pe_enable
        Enable phase encoding (useful for calibration scans if False).
    spoiling_enable
        Enable spoiling gradients.
    mrd_header_file
        Filename of the ISMRMRD header file to be created. If None, no header file is created.

    Returns
    -------
    seq
        PyPulseq Sequence object
    min_te
        Shortest possible echo time.
    min_tr
        Shortest possible repetition time.

    """
    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # create EpiReadout object
    epi2d = EpiReadout(
        system=system,
        fov=fov,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        bandwidth=bandwidth,
        oversampling=oversampling,
        readout_type=readout_type,
        ramp_sampling=ramp_sampling,
        partial_fourier_factor=partial_fourier_factor,
        pe_enable=pe_enable,
        spoiling_enable=spoiling_enable,
    )

    # create slice selective excitation pulse and gradients
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(rf_flip_angle),
        duration=rf_duration,
        slice_thickness=slice_thickness,
        apodization=rf_apodization,
        time_bw_product=rf_bwt,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
        use='excitation',
    )

    # create refocussing pulse and gradients if echo type is 'SE (spin echo)
    if echo_type == 'SE':
        rf180, gz180, _ = pp.make_sinc_pulse(
            flip_angle=np.pi,
            system=system,
            duration=rf_duration * 2,
            slice_thickness=slice_thickness,
            apodization=0.5,
            time_bw_product=4,
            phase_offset=np.pi / 2,
            use='refocusing',
            return_gz=True,
            delay=system.rf_dead_time,
        )
        _, gzr1_t, gzr1_a = pp.make_extended_trapezoid_area(
            channel='z',
            grad_start=0,
            grad_end=gz180.amplitude,
            area=1.5 * gz.area,
            system=system,
        )
        _, gzr2_t, gzr2_a = pp.make_extended_trapezoid_area(
            channel='z',
            grad_start=gz180.amplitude,
            grad_end=0,
            area=-gzr.area + 1.5 * gz.area,
            system=system,
        )

        # create combined gradient including pre/post spoiler
        gz180n = pp.make_extended_trapezoid(
            channel='z',
            system=system,
            times=np.array([*gzr1_t, *gzr2_t + gzr1_t[3] + gz180.flat_time]),
            amplitudes=np.array([*gzr1_a, *gzr2_a]),
        )

        # update rf delay of refocussing pulse to ensure it's centered in plateau of combined gradient
        rf180.delay = gzr1_t[-1]

    # calculate minimum echo time
    gzr_prephaser_dur = pp.calc_duration(gzr, epi2d.gx_pre, epi2d.gx_pre)

    # calculate echo time delay(s)
    if echo_type == 'FID':
        min_te = rf.shape_dur / 2  # time from center to end of RF pulse
        min_te += max(rf.ringdown_time, gz.fall_time)  # RF ringdown time or gradient fall time
        if add_navigator_acq:
            min_te += pp.calc_duration(gzr, epi2d.gx)
            min_te += 3 * pp.calc_duration(epi2d.gx)
            min_te += pp.calc_duration(epi2d.gy_pre)
        else:
            min_te += gzr_prephaser_dur  # for minimum TE, gzr and pre-phasers are played out simultaneously
        min_te += epi2d.time_to_center_without_prephaser

        if te is None:
            te_delay = 0.0
        else:
            te_delay = round_to_raster(te - min_te, system.block_duration_raster)
            if te_delay < 0:
                raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')
    elif echo_type == 'SE':
        t_mid_ref_to_center = rf180.shape_dur / 2
        t_mid_ref_to_center += gzr2_t[-1]
        if add_navigator_acq:
            t_mid_ref_to_center += pp.calc_duration(epi2d.gx)
            t_mid_ref_to_center += 3 * pp.calc_duration(epi2d.gx)
            t_mid_ref_to_center += pp.calc_duration(epi2d.gy_pre)
        else:
            t_mid_ref_to_center += gzr_prephaser_dur
        t_mid_ref_to_center += epi2d.time_to_center_without_prephaser

        min_te = 2 * t_mid_ref_to_center

        te_delay2 = 0.0
        te_delay1 = t_mid_ref_to_center
        te_delay1 -= rf.shape_dur / 2
        te_delay1 -= max(rf.ringdown_time, gz.fall_time)
        te_delay1 -= gzr1_t[-1]
        te_delay1 -= rf180.shape_dur / 2
        te_delay1 = round_to_raster(te_delay1, system.block_duration_raster)

        if te is not None:
            if te > min_te:
                additional_te_delay_half = round_to_raster((te - min_te) / 2, system.block_duration_raster)
                te_delay1 += additional_te_delay_half
                te_delay2 += additional_te_delay_half
            else:
                raise ValueError(
                    f'Desired TE ({te * 1000:.3f} ms) is smaller than minimum TE ({min_te * 1000:.3f} ms).'
                )

    # calculate repetition time delay (tr_delay) for current TE settings
    if echo_type == 'FID':
        min_tr = pp.calc_duration(rf, gz)
        if add_navigator_acq:
            min_tr += pp.calc_duration(gzr, epi2d.gx)
            min_tr += 3 * pp.calc_duration(epi2d.gx)
            min_tr += pp.calc_duration(epi2d.gy_pre)
        else:
            min_tr += gzr_prephaser_dur
        min_tr += epi2d.total_duration_without_prephaser
        min_tr += te_delay
    else:
        min_tr = pp.calc_duration(rf, gz)
        min_tr += te_delay1
        min_tr += pp.calc_duration(rf180, gz180n)
        min_tr += te_delay2
        if add_navigator_acq:
            min_tr += pp.calc_duration(gzr, epi2d.gx)
            min_tr += 3 * pp.calc_duration(epi2d.gx)
            min_tr += pp.calc_duration(epi2d.gy_pre)
        else:
            min_tr += gzr_prephaser_dur
        min_tr += epi2d.total_duration_without_prephaser

    if tr is None:
        tr_delay = 0.0
    else:
        tr_delay = round_to_raster(tr - min_tr, system.block_duration_raster)
        if tr_delay < 0:
            raise ValueError(f'TR must be larger than {min_tr * 1000:.2f} ms. Current value is {tr * 1000:.3f} ms.')

    # print(f'\nCurrent echo time = {(min_te + te_delay) * 1000:.3f} ms')
    # print(f'Current repetition time = {(min_tr + tr_delay) * 1000:.3f} ms')

    # create header
    if mrd_header_file:
        hdr = create_header(
            traj_type='other',
            encoding_fov=Fov(x=fov * oversampling, y=fov, z=slice_thickness),
            recon_fov=Fov(x=fov, y=fov, z=slice_thickness),
            encoding_matrix=MatrixSize(n_x=n_readout, n_y=epi2d.n_phase_encoding, n_z=1),
            recon_matrix=MatrixSize(n_x=n_readout, n_y=n_phase_encoding, n_z=1),
            dwell_time=epi2d.adc.dwell,
            slice_limits=Limits(min=0, max=n_slices, center=n_slices // 2),
            k1_limits=Limits(min=0, max=epi2d.n_phase_enc_total, center=epi2d.n_phase_enc_pre_center + 1),
            k2_limits=Limits(min=0, max=1, center=0),
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    # obtain noise samples if selected
    if add_noise_acq:
        seq.add_block(
            pp.make_label(label='LIN', type='SET', value=0),
            pp.make_label(label='SLC', type='SET', value=0),
            pp.make_label(label='NOISE', type='SET', value=True),
        )
        seq.add_block(
            epi2d.adc, pp.make_delay(round_to_raster(pp.calc_duration(epi2d.adc), system.block_duration_raster, 'ceil'))
        )
        seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
        seq.add_block(pp.make_delay(system.rf_dead_time))

        if mrd_header_file:
            acq = ismrmrd.Acquisition()
            acq.resize(trajectory_dimensions=2, number_of_samples=epi2d.adc.num_samples)
            prot.append_acquisition(acq)

    t_after_noise = sum(seq.block_durations.values())

    for slice_ in range(n_slices):
        # define label(s)
        slice_label = pp.make_label(label='SLC', type='SET', value=slice_)

        # set frequency offset for current slice
        rf.freq_offset = gz.amplitude * slice_thickness * (slice_ - (n_slices - 1) / 2)

        # add slice selective excitation pulse and set slice label
        seq.add_block(rf, gz, slice_label)

        if echo_type == 'FID' and te_delay > 0:
            seq.add_block(pp.make_delay(te_delay))
        elif echo_type == 'SE':
            seq.add_block(pp.make_delay(te_delay1))
            t_after_te_delay1 = sum(seq.block_durations.values())
            seq.add_block(rf180, gz180n)
            if te_delay2 > 0:
                seq.add_block(pp.make_delay(te_delay2))

        # add navigator scans for ghost correction
        if add_navigator_acq:
            # reverse the readout gradient in advance for navigator
            gx_pre = pp.scale_grad(epi2d.gx_pre, -1)
            gx = pp.scale_grad(epi2d.gx, -1)
            block_content = [
                gx_pre,
                pp.make_label(label='NAV', type='SET', value=1),
                pp.make_label(label='LIN', type='SET', value=floor(n_phase_encoding / 2)),
            ]
            if echo_type == 'FID':
                block_content.append(gzr)  # gzr already included in gz180n for SE
            seq.add_block(*block_content)

            # reverse gx_pre back after addBlock
            gx_pre = pp.scale_grad(gx_pre, -1)
            for n in range(3):
                seq.add_block(
                    gx,
                    epi2d.adc,
                    pp.make_label(label='REV', type='SET', value=gx.amplitude < 0),
                    pp.make_label(label='SEG', type='SET', value=gx.amplitude < 0),
                    pp.make_label(label='AVG', type='SET', value=(n + 1) == 3),
                )
                gx = pp.scale_grad(gx, -1)
                if mrd_header_file:
                    acq = ismrmrd.Acquisition()
                    acq.resize(trajectory_dimensions=2, number_of_samples=epi2d.adc.num_samples)
                    prot.append_acquisition(acq)

            # add gy_pre and reset labels
            seq.add_block(
                epi2d.gy_pre,
                pp.make_label(label='NAV', type='SET', value=0),
                pp.make_label(label='AVG', type='SET', value=0),
            )
        else:
            if echo_type == 'FID':
                gzr, gx_pre, gy_pre = pp.align(left=[gzr], right=[epi2d.gx_pre, epi2d.gy_pre])
                seq.add_block(gzr, gx_pre, gy_pre)
            else:
                gx_pre, gy_pre = pp.align(right=[epi2d.gx_pre, epi2d.gy_pre])
                seq.add_block(gx_pre, gy_pre)

        t_before_readout = sum(seq.block_durations.values())
        # add EPI readout block without pre-phaser gradients
        seq, prot = epi2d.add_to_seq(seq, add_prephaser=False, mrd_dataset=prot)

        # add repetition time delay
        if tr_delay > 0:
            seq.add_block(pp.make_delay(tr_delay))

    # calculate k-space trajectory from Sequence to add this info to mrd file
    k_traj_adc, _, _, _, _ = seq.calculate_kspace()
    samples_per_acq = epi2d.adc.num_samples
    number_of_total_acq = k_traj_adc.shape[-1] // samples_per_acq
    number_of_epi_acq = epi2d.n_phase_enc_total
    if 'data' in prot._dataset:
        number_of_noise_acq = prot.number_of_acquisitions()
    else:
        number_of_noise_acq = 0

    if not number_of_epi_acq + number_of_noise_acq == number_of_total_acq:
        raise (ValueError('Number of calculated acquisitions does not match expected number.'))

    k_traj_adc_readout = k_traj_adc[:, number_of_noise_acq * samples_per_acq :]

    # create mrd acquisition and add trajectory info for each EPI readout
    for n in range(number_of_epi_acq):
        start = n * samples_per_acq
        end = start + samples_per_acq

        traj = np.zeros((samples_per_acq, 2), dtype=np.float32)
        traj[:, 0] = np.round(k_traj_adc_readout[0, start:end] * fov, 3)
        traj[:, 1] = np.round(k_traj_adc_readout[1, start:end] * fov, 3)

        acq = ismrmrd.Acquisition()
        acq.resize(trajectory_dimensions=2, number_of_samples=samples_per_acq)
        acq.traj[:] = traj

        prot.append_acquisition(acq)

    # close ISMRMRD file
    if mrd_header_file:
        prot.close()

    # set gridding definitions extracted from EpiReadout
    if ramp_sampling:
        gridding_params = [
            epi2d.gx.rise_time,
            epi2d.gx.flat_time,
            epi2d.gx.fall_time,
            epi2d.adc.delay - epi2d.gx.delay,
            epi2d.adc.num_samples * epi2d.adc.dwell,
        ]

        seq.set_definition(key='TargetGriddedSamples', value=n_readout * oversampling)
        seq.set_definition(key='TrapezoidGriddingParameters', value=gridding_params)

    # write all required parameters in the seq-file header/definitions
    seq.set_definition('FOV', [fov, fov, slice_thickness * n_slices])
    seq.set_definition('ReconMatrix', (n_readout, n_phase_encoding, n_slices))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('TE', te or min_te)
    seq.set_definition('TR', tr or min_tr)
    seq.set_definition('ReadoutOversamplingFactor', oversampling)

    t_exc = t_after_noise + rf.delay + rf.shape_dur / 2
    if echo_type == 'SE':
        t_ref = t_after_te_delay1 + rf180.delay + rf180.shape_dur / 2
    else:
        t_ref = 0.0
    t_echo = t_before_readout + epi2d.time_to_center_without_prephaser

    return seq, min_te, min_tr, t_exc, t_ref, t_echo


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr: float | None = None,
    fov: float = 200e-3,
    n_readout: int = 16,
    n_phase_encoding: int = 16,
    n_slices: int = 1,
    slice_thickness: float = 4e-3,
    bandwidth: float = 64e3,
    readout_type: Literal['symmetric', 'flyback'] = 'symmetric',
    echo_type: Literal['FID', 'SE'] = 'FID',
    oversampling: Literal[1, 2, 4] = 2,
    ramp_sampling: bool = False,
    partial_fourier_factor: float = 1,
    add_navigator_acq: bool = True,
    add_noise_acq: bool = True,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate an Echo-Planar Imaging (EPI) sequence.

    Returns
    -------
    seq
        Sequence object of radial FLASH sequence.
    file_path
        Path to the sequence file.
    """
    if system is None:
        system = sys_defaults

    # define settings of rf excitation pulse
    rf_flip_angle = 90.0  # flip angle of the rf excitation pulse [°]
    rf_duration = 1.28e-3  # duration of the rf excitation pulse [s]
    rf_bwt = 4.0  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse

    # define EPI settings
    enable_phase_encoding = True
    enable_gradient_spoiling = True

    # define sequence filename
    rs_string = 'rs' if ramp_sampling else 'nors'
    pf_string = f'{partial_fourier_factor}pf'.replace('.', 'p')

    filename = f'{Path(__file__).stem}_{int(fov * 1000)}fov_{n_readout}nx_{n_phase_encoding}ny'
    filename += f'_{readout_type}_{echo_type}_{oversampling}ro_{rs_string}_{pf_string}_with_nav'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    mrd_file = output_path / Path(filename + '_header.h5')

    seq, _min_te, min_tr, t_exc, t_ref, t_echo = epi2d_kernel(
        system=system,
        te=te,
        tr=tr,
        fov=fov,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        bandwidth=bandwidth,
        oversampling=oversampling,
        slice_thickness=slice_thickness,
        n_slices=n_slices,
        rf_duration=rf_duration,
        rf_flip_angle=rf_flip_angle,
        rf_bwt=rf_bwt,
        rf_apodization=rf_apodization,
        readout_type=readout_type,
        echo_type=echo_type,
        ramp_sampling=ramp_sampling,
        partial_fourier_factor=partial_fourier_factor,
        pe_enable=enable_phase_encoding,
        spoiling_enable=enable_gradient_spoiling,
        add_noise_acq=add_noise_acq,
        add_navigator_acq=add_navigator_acq,
        mrd_header_file=mrd_file,
    )

    print(f'Time from exc to ref: {(t_ref - t_exc) * 1000}')
    print(f'Time from ref to echo: {(t_echo - t_ref) * 1000}')
    print(f'Time from exc to echo: {(t_echo - t_exc) * 1000}')

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

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    if show_plots:
        fig1, axs1, fig2, axs2 = seq.plot(time_range=(0, 10 * (tr or min_tr)), plot_now=False)
        for ax in [*axs1, *axs2]:
            ax.axvline(t_exc, color='red')
            ax.axvline(t_ref, color='red')
            ax.axvline(t_echo, color='red')
        plt.show()

    return seq, output_path / filename


if __name__ == '__main__':
    main()
