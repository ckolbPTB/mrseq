"""2D Echo Planar Imaging (EPI) sequence."""

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
    oversampling: Literal[1, 2, 4],
    ramp_sampling: bool,
    partial_fourier_factor: float,
    pe_enable: bool,
    spoiling_enable: bool,
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

    # calculate minimum echo time
    gzr_prephaser_dur = pp.calc_duration(gzr, epi2d.gx_pre, epi2d.gx_pre)
    min_te = rf.shape_dur / 2  # time from center to end of RF pulse
    min_te += max(rf.ringdown_time, gz.fall_time)  # RF ringdown time or gradient fall time
    min_te += gzr_prephaser_dur  # for minimum TE, gzr and pre-phasers are played out simultaneously
    min_te += epi2d.time_to_center_without_prephaser

    # calculate echo time delay (te_delay)
    if te is None:
        te_delay = 0.0
    else:
        te_delay = round_to_raster(te - min_te, system.block_duration_raster)
        if te_delay < 0:
            raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')

    # calculate minimum repetition time depending on chosen echo time
    min_tr = pp.calc_duration(rf, gz)
    min_tr += gzr_prephaser_dur  # if TE is chosen minimal, gzr and pre-phasers are played out simultaneously
    min_tr += epi2d.total_duration_without_prephaser

    # calculate repetition time delay (tr_delay) for current TE settings
    current_min_tr = min_tr + te_delay
    if tr is None:
        tr_delay = 0.0
    else:
        tr_delay = round_to_raster(tr - current_min_tr, system.block_duration_raster)
        if tr_delay < 0:
            raise ValueError(
                f'TR must be larger than {current_min_tr * 1000:.2f} ms. Current value is {tr * 1000:.3f} ms.'
            )

    print(f'\nCurrent echo time = {(min_te + te_delay) * 1000:.3f} ms')
    print(f'Current repetition time = {(current_min_tr + tr_delay) * 1000:.3f} ms')

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

    # obtain noise samples
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
        # define label(s)
        slice_label = pp.make_label(label='SLC', type='SET', value=slice_)

        # set frequency offset for current slice
        rf.freq_offset = gz.amplitude * slice_thickness * (slice_ - (n_slices - 1) / 2)

        # add slice selective excitation pulse and set slice label
        seq.add_block(rf, gz, slice_label)

        # add echo time delay
        if te_delay > 0:
            seq.add_block(pp.make_delay(te_delay))

        # add slice selection rephaser and readout prephaser gradients
        gzr, gx_pre, gy_pre = pp.align(left=[gzr], right=[epi2d.gx_pre, epi2d.gy_pre])
        seq.add_block(gzr, gx_pre, gy_pre)

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

    return seq, min_te, min_tr


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr: float | None = None,
    fov: float = 256e-3,
    n_readout: int = 64,
    n_phase_encoding: int = 64,
    n_slices: int = 1,
    slice_thickness: float = 8e-3,
    bandwidth: float = 64e3,
    readout_type: Literal['symmetric', 'flyback'] = 'flyback',
    oversampling: Literal[1, 2, 4] = 1,
    ramp_sampling: bool = False,
    partial_fourier_factor: float = 1,
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
    rf_flip_angle: float = 90  # flip angle of the rf excitation pulse [°]
    rf_duration = 2.56e-3  # duration of the rf excitation pulse [s]
    rf_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse

    # define EPI settings
    enable_phase_encoding = True
    enable_gradient_spoiling = True

    # define sequence filename
    rs_string = 'rs' if ramp_sampling else 'nors'
    pf_string = f'{partial_fourier_factor}'.replace('.', 'p')

    filename = f'{Path(__file__).stem}_{int(fov * 1000)}fov_{n_readout}nx_{n_phase_encoding}ny'
    filename += f'_{readout_type}_{oversampling}ro_{rs_string}_{pf_string}'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    mrd_file = output_path / Path(filename + '_header.h5')

    seq, _min_te, min_tr = epi2d_kernel(
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
        pe_enable=enable_phase_encoding,
        spoiling_enable=enable_gradient_spoiling,
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

    # show advanced rest report
    if test_report:
        print('\nCreating advanced test report...')
        print(seq.test_report())

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    if show_plots:
        seq.plot(time_range=(0, 10 * (tr or min_tr)))

    return seq, output_path / filename


if __name__ == '__main__':
    main()
