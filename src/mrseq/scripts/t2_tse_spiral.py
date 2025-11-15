"""Spiral turbo-spin echo sequence for T2 mapping."""

from pathlib import Path
from typing import Literal

import ismrmrd
import numpy as np
import pypulseq as pp

from mrseq.utils import round_to_raster
from mrseq.utils import spiral_acquisition
from mrseq.utils import sys_defaults
from mrseq.utils.ismrmrd import Fov
from mrseq.utils.ismrmrd import Limits
from mrseq.utils.ismrmrd import MatrixSize
from mrseq.utils.ismrmrd import create_header


def t2_tse_spiral_kernel(
    system: pp.Opts,
    te: float | None,
    n_echoes: int,
    tr: float,
    fov_xy: float,
    fov_z: float,
    n_readout: int,
    readout_oversampling: Literal[1, 2, 4],
    spiral_undersampling: float,
    n_slice_encoding: int,
    gx_pre_duration: float,
    rf_ex_duration: float,
    rf_ex_bwt: float,
    rf_ref_duration: float,
    rf_ref_bwt: float,
    rf_ref_width_scale_factor: float,
    gz_crusher_duration: float,
    gz_crusher_area: float,
    mrd_header_file: str | None,
) -> tuple[pp.Sequence, float]:
    """Generate a spiral TSE sequence for T2-mapping.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) in seconds. Minimum echo time is used if set to None.
    n_echoes
        Number of echoes in one TSE train.
    tr
        Desired repetition time (TR) in seconds.
    fov_xy
        Field of view in the x and y directions in meters.
    fov_z
        Field of view in the z direction (slice thickness) in meters.
    n_readout
        Number of frequency encoding steps.
    readout_oversampling
        Readout oversampling. Determines the number of ADC samples along a spiral and the bandwidth.
    spiral_undersampling
        Angular undersampling of the spiral trajectoy.
    n_slice_encoding
        Number of slice encoding steps.
    gx_pre_duration
        Duration of the readout pre-winder gradient in seconds..
    rf_ex_duration
        Duration of the excitation RF pulse in seconds.
    rf_ex_bwt
        Bandwidth-time product of the excitation RF pulse.
    rf_ref_duration
        Duration of the refocusing RF pulse in seconds.
    rf_ref_bwt
        Bandwidth-time product of the refocusing RF pulse.
    rf_ref_width_scale_factor
        Factor to scale the slice thickness of the refocusing pulse.
    readout_oversampling
        Readout oversampling.
    gz_crusher_duration
        Duration of the crusher gradients applied around the 180° pulse.
    gz_crusher_area : float
        Area (zeroth gradient moment) of the crusher gradients applied around the 180° pulse.
    mrd_header_file
        Filename of the ISMRMRD header file to be created. If None, no header file is created.

    Returns
    -------
    seq
        PyPulseq Sequence object
    min_te
        Shortest possible echo time.
    """
    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # create slice selective excitation pulse and gradient
    rf_ex, gz_ex, gzr_ex = pp.make_sinc_pulse(
        flip_angle=90 / 180 * np.pi,
        duration=rf_ex_duration,
        slice_thickness=fov_z,
        apodization=0.5,
        phase_offset=np.pi / 2,
        time_bw_product=rf_ex_bwt,
        delay=system.rf_dead_time,  # delay should equal at least the dead time of the RF pulse
        system=system,
        return_gz=True,
        use='excitation',
    )

    # create slice selective refocusing pulse and gradient
    rf_ref, gz_ref, _ = pp.make_sinc_pulse(
        flip_angle=np.pi,
        duration=rf_ref_duration,
        slice_thickness=fov_z * rf_ref_width_scale_factor,
        apodization=0.5,
        phase_offset=0.0,
        time_bw_product=rf_ref_bwt,
        delay=system.rf_dead_time,  # delay should equal at least the dead time of the RF pulse
        system=system,
        return_gz=True,
        use='refocusing',
    )

    # create readout gradient and ADC
    gx, gy, adc, trajectory, time_to_echo = spiral_acquisition(
        system,
        n_readout,
        fov_xy,
        spiral_undersampling,
        readout_oversampling=readout_oversampling,
        n_spirals=None,
        max_pre_duration=gx_pre_duration,
        spiral_type='in-out',
    )

    # phase encoding along slice direction
    gz_areas = (np.arange(n_slice_encoding) - n_slice_encoding // 2) * 1 / fov_z

    # create crusher gradients
    gz_crush = pp.make_trapezoid(channel='z', system=system, area=gz_crusher_area, duration=gz_crusher_duration)

    # calculate minimum delays
    # tau1: between excitation pulse and first refocusing pulse
    min_tau1 = rf_ex.shape_dur / 2
    min_tau1 += max(rf_ex.ringdown_time, gz_ex.fall_time)
    min_tau1 += pp.calc_duration(gzr_ex)
    min_tau1 += max(rf_ref.delay, gz_ref.delay + gz_ref.rise_time)
    min_tau1 += rf_ref.shape_dur / 2
    min_tau1 += pp.calc_duration(gz_crush)

    # tau2: between refocusing pulses and readout
    min_tau2 = rf_ref.shape_dur / 2
    min_tau2 += max(rf_ref.ringdown_time, gz_ref.fall_time)
    min_tau2 += time_to_echo
    min_tau2 += pp.calc_duration(gz_crush)

    # tau3: between readout and next refocusing pulse
    min_tau3 = time_to_echo  # TODO double check
    min_tau3 += rf_ref.shape_dur / 2
    min_tau3 += max(rf_ref.delay, gz_ref.delay + gz_ref.rise_time)
    min_tau3 += pp.calc_duration(gz_crush)

    min_te = (
        2
        * max(
            round_to_raster(min_tau1, system.block_duration_raster),
            round_to_raster(min_tau2, system.block_duration_raster),
            round_to_raster(min_tau3, system.block_duration_raster),
        ).item()
    )

    # calculate echo time delay (te_delay)
    te = min_te if te is None else round_to_raster(te, system.block_duration_raster)
    if te < min_te:
        raise ValueError(f'TE must be larger than {min_te * 1000:.2f} ms. Current value is {te * 1000:.2f} ms.')

    tau1 = round_to_raster(te / 2 - min_tau1, raster_time=system.grad_raster_time)
    tau2 = round_to_raster(te / 2 - min_tau2, raster_time=system.grad_raster_time)
    tau3 = round_to_raster(te / 2 - min_tau3, raster_time=system.grad_raster_time)
    print(f'\nCurrent echo time = {(te) * 1000:.2f} ms')

    # create header
    if mrd_header_file:
        hdr = create_header(
            traj_type='other',
            encoding_fov=Fov(x=fov_xy, y=fov_xy, z=fov_z),
            recon_fov=Fov(x=fov_xy, y=fov_xy, z=fov_z),
            encoding_matrix=MatrixSize(n_x=int(n_readout), n_y=int(n_readout), n_z=n_slice_encoding),
            recon_matrix=MatrixSize(n_x=n_readout, n_y=n_readout, n_z=n_slice_encoding),
            dwell_time=adc.dwell,
            k1_limits=Limits(min=0, max=len(gx), center=0),
            k2_limits=Limits(min=0, max=n_slice_encoding, center=n_slice_encoding // 2),
            slice_limits=Limits(),
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    # obtain noise samples
    seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='SLC', type='SET', value=0))
    seq.add_block(adc, pp.make_label(label='NOISE', type='SET', value=True))
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    if mrd_header_file:
        acq = ismrmrd.Acquisition()
        acq.resize(trajectory_dimensions=3, number_of_samples=adc.num_samples)
        prot.append_acquisition(acq)

    # add all events to the sequence
    for se in range(n_slice_encoding):
        se_label = pp.make_label(type='SET', label='PAR', value=int(se))

        # phase encoding along se
        gz_pre = pp.make_trapezoid(
            channel='z',
            area=gz_areas[se],
            duration=gx_pre_duration,
            system=system,
        )

        for spiral_ in range(len(gx)):
            pe_label = pp.make_label(type='SET', label='LIN', value=spiral_)

            _start_time_tr_block = sum(seq.block_durations.values())

            # add excitation pulse
            seq.add_block(rf_ex, gz_ex)
            seq.add_block(gzr_ex)
            seq.add_block(pp.make_delay(tau1))

            for echo in range(n_echoes):
                echo_label = pp.make_label(type='SET', label='ECO', value=int(echo))

                # add refocusing pulse with crusher gradients
                seq.add_block(gz_crush)
                seq.add_block(rf_ref, gz_ref)
                seq.add_block(gz_crush)

                seq.add_block(pp.make_delay(tau2))

                # add pre gradients and all labels
                labels = [se_label, pe_label, echo_label]
                seq.add_block(gx[spiral_], gy[spiral_], gz_pre, adc, *labels)

                # rewind gradients
                seq.add_block(pp.scale_grad(gz_pre, -1))

                if echo < n_echoes - 1:
                    seq.add_block(pp.make_delay(tau3))

                if mrd_header_file and spiral_ >= 0:
                    # add acquisitions to metadata
                    spiral_trajectory = np.zeros((trajectory.shape[1], 3), dtype=np.float32)

                    # the spiral trajectory is calculated in units of delta_k.
                    # For image reconstruction we use delta_k = 1
                    spiral_trajectory[:, 0] = trajectory[spiral_, :, 0] * fov_xy
                    spiral_trajectory[:, 1] = trajectory[spiral_, :, 1] * fov_xy
                    spiral_trajectory[:, 2] = se - n_slice_encoding // 2

                    acq = ismrmrd.Acquisition()
                    acq.resize(trajectory_dimensions=3, number_of_samples=adc.num_samples)
                    acq.traj[:] = spiral_trajectory
                    prot.append_acquisition(acq)

            duration_tr_block = sum(seq.block_durations.values()) - _start_time_tr_block
            tr_delay = round_to_raster(tr - duration_tr_block, system.block_duration_raster)
            if tr_delay < 0:
                raise ValueError('Desired TR too short for given sequence parameters.')
            seq.add_block(pp.make_delay(tr_delay))

    return seq, min_te


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    n_echoes: int = 10,
    tr: float = 4,
    fov_xy: float = 128e-3,
    fov_z: float = 80e-3,
    n_readout: int = 128,
    readout_oversampling: Literal[1, 2, 4] = 2,
    n_spiral_arms: int = 128,
    n_slice_encoding=10,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
) -> pp.Sequence:
    """Generate spiral TSE sequence for T2-mapping.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    n_echoes
        Number of echoes in one TSE train.
    tr
        Desired repetition time (TR) (in seconds).
    fov_xy
        Field of view in x and y direction (in meters).
    fov_z
        Field of view along z (in meters).
    n_readout
        Number of frequency encoding steps.
    readout_oversampling
        Readout oversampling. Determines the number of ADC samples along a spiral and the bandwidth.
    n_spiral_arms
        Number of spiral arms.
    n_slice_encoding
        Number of phase encoding steps along the slice direction.
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
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]

    gz_crusher_duration = 1.6e-3  # duration of crusher gradients [s]
    gz_crusher_area = 4 / (fov_z / n_slice_encoding)

    # define settings of rf excitation pulse
    rf_ex_duration = 2e-3  # duration of the rf excitation pulse [s]
    rf_ex_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]

    rf_ref_width_scale_factor = 3.5  # width of refocusing pulse is increased compared to excitation pulse

    # define sequence filename
    filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_xy_{int(fov_z * 1000)}_fov_z_'
    filename += f'{n_readout}nx_{n_spiral_arms}na_{n_slice_encoding}nz'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    seq, min_te = t2_tse_spiral_kernel(
        system=system,
        te=te,
        n_echoes=n_echoes,
        tr=tr,
        fov_xy=fov_xy,
        fov_z=fov_z,
        n_readout=n_readout,
        readout_oversampling=readout_oversampling,
        spiral_undersampling=n_readout / n_spiral_arms,
        n_slice_encoding=n_slice_encoding,
        gx_pre_duration=gx_pre_duration,
        rf_ex_duration=rf_ex_duration,
        rf_ex_bwt=rf_ex_bwt,
        rf_ref_duration=rf_ex_duration * 2,
        rf_ref_bwt=rf_ex_bwt,
        rf_ref_width_scale_factor=rf_ref_width_scale_factor,
        gz_crusher_duration=gz_crusher_duration,
        gz_crusher_area=gz_crusher_area,
        mrd_header_file=output_path / Path(filename + '_header.h5'),
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
    seq.set_definition('FOV', [fov_xy, fov_xy, fov_z])
    seq.set_definition('ReconMatrix', (n_readout, n_readout, 1))
    te_list = np.cumsum((te,) * n_echoes if te else (min_te,) * n_echoes)
    seq.set_definition('TE', te_list.tolist())
    seq.set_definition('TR', tr)

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    seq.write(str(output_path / filename), create_signature=True)

    if show_plots:
        seq.plot(time_range=(0, te_list[-1] * 1.2))

    return seq, output_path / filename


if __name__ == '__main__':
    main()
