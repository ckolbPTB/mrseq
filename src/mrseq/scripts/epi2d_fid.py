"""2D Echo Planar Imaging (EPI) FID sequence."""

from math import floor
from pathlib import Path
from typing import Literal

import ismrmrd
import numpy as np
import pypulseq as pp

from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils.ismrmrd import Fov
from mrseq.utils.ismrmrd import Limits
from mrseq.utils.ismrmrd import MatrixSize
from mrseq.utils.ismrmrd import create_header
from mrseq.utils.trajectory import EpiReadout


def epi2d_fid_kernel(
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
    oversampling: Literal[1, 2, 4],
    ramp_sampling: bool,
    partial_fourier_factor: float,
    add_spoiler: bool,
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
        Total receiver bandwidth (in Hz).
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    n_slices
        Number of slices.
    rf_duration
        Duration of the rf excitation pulse (in seconds)
    rf_flip_angle
        Flip angle of rf excitation pulse (in degrees)
    rf_bwt
        Bandwidth-time product of rf excitation pulse (in Hz * seconds)
    rf_apodization
        Apodization factor of rf excitation pulse.
    readout_type
        Readout type ('symmetric' or 'flyback').
    oversampling
        Readout oversampling factor. Can be 1 (no oversampling), 2, or 4.
    ramp_sampling
        If True, ADC is active during gradient ramps for optimized timing.
    partial_fourier_factor
        Desired partial Fourier factor in "phase encoding" direction. Must be larger than 0.5 and smaller or equal to 1.
        The actual partial Fourier factor might slightly deviate from the desired value.
    add_spoiler
        If True, a spoiler gradients will be added to the sequence after the EPI readout.
    add_noise_acq
        If True, noise acquisitions will be added at the beginning of the sequence.
    add_navigator_acq
        If True, 3 navigator acquisitions will be added to the sequence to allow for ghost corrections.
        The navigator acquisitions are added between the rf excitation pulse and the EPI readout.
        Be aware that navigator acquisitions will increase the minimum echo and repetition times.
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
        pe_enable=True,
        spoiling_enable=add_spoiler,
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

    # calculate echo time delay
    min_te = rf.shape_dur / 2  # time from center to end of RF pulse
    min_te += max(rf.ringdown_time, gz.fall_time)  # RF ringdown time or gradient fall time
    if add_navigator_acq:
        min_te += pp.calc_duration(gzr, epi2d.gx_pre)
        min_te += 3 * pp.calc_duration(epi2d.gx)
        min_te += pp.calc_duration(epi2d.gy_pre)
    else:
        min_te += pp.calc_duration(gzr, epi2d.gx_pre, epi2d.gx_pre)
    min_te += epi2d.time_to_center_without_prephaser

    if te is None:
        te_delay = 0.0
    else:
        te_delay = round_to_raster(te - min_te, system.block_duration_raster)
        if te_delay < 0:
            raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')

    # calculate repetition time delay (tr_delay) for current TE settings
    min_tr = pp.calc_duration(rf, gz)
    if add_navigator_acq:
        min_tr += pp.calc_duration(gzr, epi2d.gx_pre)
        min_tr += 3 * pp.calc_duration(epi2d.gx)
        min_tr += pp.calc_duration(epi2d.gy_pre)
    else:
        min_tr += pp.calc_duration(gzr, epi2d.gx_pre, epi2d.gx_pre)
    min_tr += epi2d.total_duration_without_prephaser
    min_tr += te_delay

    if tr is None:
        tr_delay = 0.0
    else:
        tr_delay = round_to_raster(tr - min_tr, system.block_duration_raster)
        if tr_delay < 0:
            raise ValueError(f'TR must be larger than {min_tr * 1000:.2f} ms. Current value is {tr * 1000:.3f} ms.')

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

    for slice_ in range(n_slices):
        # define slice label
        slice_label = pp.make_label(label='SLC', type='SET', value=slice_)

        # set frequency offset for current slice
        rf.freq_offset = gz.amplitude * slice_thickness * (slice_ - (n_slices - 1) / 2)
        # rf.phase_offset = - 2 * np.pi * rf.freq_offset * pp.calc_rf_center(rf)

        # add slice-selective excitation pulse and set slice label
        seq.add_block(rf, gz, slice_label)

        # add navigator scans for ghost correction
        if add_navigator_acq:
            # reverse the readout gradient and pre-winder in advance for navigator
            gx_pre = pp.scale_grad(epi2d.gx_pre, -1)
            gx = pp.scale_grad(epi2d.gx, -1)
            # add slice selection rewinder and readout pre-winder in x direction (gy_pre will be added after navigators)
            gzr, gx_pre = pp.align(left=[gzr], right=[gx_pre])
            seq.add_block(
                gzr,
                gx_pre,
                pp.make_label(label='NAV', type='SET', value=1),
                pp.make_label(label='LIN', type='SET', value=floor(n_phase_encoding / 2)),
            )
            # reverse gx_pre back after adding to sequence
            gx_pre = pp.scale_grad(gx_pre, -1)

            # add 3 navigator acquisitions
            for n in range(3):
                seq.add_block(
                    gx,
                    epi2d.adc,
                    pp.make_label(label='REV', type='SET', value=gx.amplitude < 0),
                    pp.make_label(label='SEG', type='SET', value=gx.amplitude < 0),
                    pp.make_label(label='AVG', type='SET', value=(n + 1) == 3),
                )
                gx = pp.scale_grad(gx, -1)
                # add navigator acquisitions to ISMRMRD file
                if mrd_header_file:
                    acq = ismrmrd.Acquisition()
                    acq.resize(trajectory_dimensions=2, number_of_samples=epi2d.adc.num_samples)
                    prot.append_acquisition(acq)

            # add echo time delay
            seq.add_block(pp.make_delay(te_delay))

            # add gy_pre and reset labels
            seq.add_block(
                epi2d.gy_pre,
                pp.make_label(label='NAV', type='SET', value=0),
                pp.make_label(label='AVG', type='SET', value=0),
            )
        else:
            # add echo time delay
            seq.add_block(pp.make_delay(te_delay))

            # align and add slice-selection rewinder and readout pre-winder gradients
            gzr, gx_pre, gy_pre = pp.align(left=[gzr], right=[epi2d.gx_pre, epi2d.gy_pre])
            seq.add_block(gzr, gx_pre, gy_pre)

        # add EPI readout block without pre-phaser gradients
        seq, prot = epi2d.add_to_seq(seq, add_prephaser=False, mrd_dataset=prot)

        # add repetition time delay
        if tr_delay > 0:
            seq.add_block(pp.make_delay(tr_delay))

    if mrd_header_file:
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

    return seq, min_te, min_tr


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr: float | None = None,
    fov: float = 200e-3,
    n_readout: int = 64,
    n_phase_encoding: int = 64,
    n_slices: int = 1,
    slice_thickness: float = 8e-3,
    bandwidth: float = 100e3,
    readout_type: Literal['symmetric', 'flyback'] = 'symmetric',
    oversampling: Literal[1, 2, 4] = 2,
    ramp_sampling: bool = True,
    partial_fourier_factor: float = 0.7,
    add_navigator_acq: bool = True,
    add_noise_acq: bool = True,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate a 2D Echo Planar Imaging (EPI) FID sequence.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds). Minimum repetition time is used if set to None.
    fov
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    n_phase_encoding
        Number of phase encoding steps.
    n_slices
        Number of slices.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    bandwidth
        Total receiver bandwidth (in Hz).
    readout_type
        Readout type ('symmetric' or 'flyback').
    oversampling
        Readout oversampling factor. Can be 1 (no oversampling), 2, or 4.
    ramp_sampling
        If True, ADC is active during gradient ramps for optimized timing.
    partial_fourier_factor
        Desired partial Fourier factor in "phase encoding" direction.
    add_navigator_acq
        If True, navigator acquisitions will be added for ghost corrections.
    add_noise_acq
        If True, noise acquisitions will be added at the beginning of the sequence.
    show_plots
        Toggles sequence plot.
    test_report
        Toggles advanced test report.
    timing_check
        Toggles timing check of the sequence.

    Returns
    -------
    seq
        Sequence object of 2D EPI FID sequence.
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

    # define spoiling settings
    enable_gradient_spoiling = True

    # define sequence filename
    rs_string = 'rs' if ramp_sampling else 'nors'  # ramp sampling
    pf_string = f'{partial_fourier_factor}pf'.replace('.', 'p')  # partial fourier factor
    readout_string = 'sym' if readout_type == 'symmetric' else 'flyb'  # readout type
    noise_string = 'withnoise' if add_noise_acq else 'nonoise'  # noise acquisition
    nav_string = 'withnav' if add_navigator_acq else 'nonav'  # navigator acquisition

    filename = f'{Path(__file__).stem}_{int(fov * 1000)}fov_{n_readout}px'
    filename += f'_{readout_string}_fid_{oversampling}ro_{rs_string}_{pf_string}'
    filename += f'_{noise_string}_{nav_string}'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    mrd_file = output_path / Path(filename + '_header.h5')

    seq, _min_te, _min_tr = epi2d_fid_kernel(
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
        ramp_sampling=ramp_sampling,
        partial_fourier_factor=partial_fourier_factor,
        add_spoiler=enable_gradient_spoiling,
        add_noise_acq=add_noise_acq,
        add_navigator_acq=add_navigator_acq,
        mrd_header_file=mrd_file,
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

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    if show_plots:
        seq.plot(time_range=(0, tr or _min_tr), plot_now=True)

    return seq, output_path / filename


if __name__ == '__main__':
    main()
