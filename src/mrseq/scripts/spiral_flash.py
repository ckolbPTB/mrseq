"""M2D spiral FLASH sequence."""

from pathlib import Path
from typing import Literal

import ismrmrd
import numpy as np
import pypulseq as pp

from mrseq.preparations import add_t1_inv_prep
from mrseq.preparations import add_t2_prep
from mrseq.preparations.receiver_gain_calibration import add_gre_receiver_gain_calibration
from mrseq.utils import round_to_raster
from mrseq.utils import spiral_acquisition
from mrseq.utils import sys_defaults
from mrseq.utils import write_sequence
from mrseq.utils.ismrmrd import Fov
from mrseq.utils.ismrmrd import Limits
from mrseq.utils.ismrmrd import MatrixSize
from mrseq.utils.ismrmrd import create_header


def _rotation_matrix(theta):
    # R[0] = (R[0][0], R[0][1], R[0][2])
    R0 = np.stack((np.cos(theta), -np.sin(theta), np.zeros_like(theta)), axis=1)  # (nangles, 3)

    # R[1] = (R[1][0], R[1][1], R[1][2])
    R1 = np.stack((np.sin(theta), np.cos(theta), np.zeros_like(theta)), axis=1)  # (nangles, 3)

    # R[2] = (R[2][0], R[2][1], R[2][2])
    R2 = np.stack((np.zeros_like(theta), np.zeros_like(theta), np.ones_like(theta)), axis=1)  # (nangles, 3)

    return np.stack((R0, R1, R2), axis=1)  # (nangles, 3, 3)


def spiral_flash_kernel(
    system: pp.Opts,
    te: float | None,
    tr: float | None,
    fov_xy: float,
    n_readout: int,
    readout_oversampling: Literal[1, 2, 4],
    spiral_undersampling: float,
    slice_thickness: float,
    n_slices: int,
    n_dummy_excitations: int,
    spiral_sampling_period: float | None,
    g_spiral_rew_slew_rate_scaling: float,
    gx_pre_duration: float,
    rf_duration: float,
    rf_flip_angle: float,
    rf_bwt: float,
    rf_apodization: float,
    rf_spoiling_phase_increment: float,
    gz_spoil_duration: float,
    gz_spoil_area: float,
    use_ext_rot: bool,
    mrd_header_file: str | Path | None,
) -> tuple[pp.Sequence, float, float]:
    """Generate a spiral FLASH sequence.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    te
        Desired echo time (TE) (in seconds). Minimum echo time is used if set to None.
    tr
        Desired repetition time (TR) (in seconds). Minimum repetition time is used if set to None.
    fov_xy
        Field of view in x and y direction (in meters).
    n_readout
        Number of frequency encoding steps.
    readout_oversampling
        Readout oversampling. Determines the number of ADC samples along a spiral and the bandwidth.
    spiral_undersampling
        Angular undersampling of the spiral trajectoy.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    n_slices
        Number of slices.
    n_dummy_excitations
        Number of dummy excitations before data acquisition to ensure steady state.
    spiral_sampling_period
        Sampling period for the readout trajectory. If None, the system gradient raster time is used.
    g_spiral_rew_slew_rate_scaling
        Scaling of max slew rate for rewinder of spiral readout gradient
    gx_pre_duration
        Duration of readout pre-winder gradient (in seconds)
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
    min_cardiac_trigger_delay = 0.12
    ge_segment_delay = 117e-6

    if n_dummy_excitations < 0:
        raise ValueError('Number of dummy excitations must be >= 0.')

    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # cMRF specific settings
    n_blocks = 15  # number of heartbeat blocks
    minimum_time_to_set_label = round_to_raster(
        1e-5, system.block_duration_raster
    )  # minimum time to set a label (in seconds)

    # create flip angle pattern
    max_flip_angles_deg = [12.5, 18.75, 25, 25, 25, 12.5, 18.75, 25, 25.0, 25, 12.5, 18.75, 25, 25, 25]
    flip_angles = np.deg2rad(
        np.concatenate(
            [
                np.concatenate((np.linspace(4, max_angle, 16), np.full((31,), max_angle)))
                for max_angle in max_flip_angles_deg
            ]
        )
    )

    # make sure the number of blocks fits the total number of flip angles / repetitions
    if not flip_angles.size % n_blocks == 0:
        raise ValueError('Number of repetitions must be a multiple of the number of blocks.')

    # calculate number of shots / repetitions per block
    n_shots_per_block = flip_angles.size // n_blocks

    # create rf dummy pulse (required for some timing calculations)
    rf_max_fa, gz_max_fa, gzr_max_fa = pp.make_sinc_pulse(
        flip_angle=np.max(flip_angles),
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
    gx, gy, adc, trajectory, time_to_echo = spiral_acquisition(
        system,
        n_readout,
        fov_xy,
        spiral_undersampling,
        readout_oversampling=readout_oversampling,
        n_spirals=None,
        max_pre_duration=gx_pre_duration,
        spiral_type='out',
        sampling_period=spiral_sampling_period,
        g_rew_slew_rate_scaling=g_spiral_rew_slew_rate_scaling,
    )
    delta_array = 2 * np.pi / len(gx) * np.arange(len(gx))  # angle difference between subsequent spirals
    max_spiral_duration = max(pp.calc_duration(gx, gy) for gx, gy in zip(gx, gy, strict=True))
    max_spiral_duration = pp.calc_duration(gx[0], gy[0])

    # create spoiler gradients
    gz_spoil = pp.make_trapezoid(channel='z', system=system, area=gz_spoil_area, duration=gz_spoil_duration)

    min_te = (
        rf_max_fa.shape_dur / 2  # time from center to end of RF pulse
        + max(rf_max_fa.ringdown_time, gz_max_fa.fall_time)  # RF ringdown time or gradient fall time
        + pp.calc_duration(gzr_max_fa)  # slice rewinder
        + time_to_echo
    )

    # calculate echo time delay (te_delay)
    if te is None:
        te_delay = 0.0
    else:
        te_delay = round_to_raster(te - min_te, system.block_duration_raster)
        if te_delay < 0:
            raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')

    # calculate minimum repetition time
    min_tr = (
        pp.calc_duration(gz_max_fa)  # rf pulse
        + pp.calc_duration(gzr_max_fa)  # slice rewinder
        + max_spiral_duration  # readout gradient
        + pp.calc_duration(gz_spoil)  # gradient spoiler or readout-re-winder
    )

    # calculate repetition time delay (tr_delay)
    current_min_tr = min_tr + te_delay
    if tr is None:
        tr_delay = 0.0
    else:
        tr_delay = round_to_raster(tr - current_min_tr, system.block_duration_raster)
        if tr_delay < 0:
            raise ValueError(
                f'TR must be larger than {current_min_tr * 1000:.3f} ms. Current value is {tr * 1000:.3f} ms.'
            )

    print(f'\nCurrent echo time = {(min_te + te_delay) * 1000:.3f} ms')
    print(f'Current repetition time = {(current_min_tr + tr_delay) * 1000:.3f} ms')

    # create header
    if mrd_header_file:
        hdr = create_header(
            traj_type='other',
            encoding_fov=Fov(x=fov_xy, y=fov_xy, z=slice_thickness),
            recon_fov=Fov(x=fov_xy, y=fov_xy, z=slice_thickness),
            encoding_matrix=MatrixSize(n_x=int(n_readout), n_y=int(n_readout), n_z=1),
            recon_matrix=MatrixSize(n_x=n_readout, n_y=n_readout, n_z=1),
            dwell_time=adc.dwell,
            slice_limits=Limits(min=0, max=n_slices, center=0),
            k1_limits=Limits(min=0, max=len(gx), center=0),
            h1_resonance_freq=system.gamma * system.B0,
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    ge_pislquant = 0
    if ge_pislquant > 0:
        n_readout_rx_gain = 128
        seq, _ = add_gre_receiver_gain_calibration(
            system=system,
            seq=seq,
            n_rep=ge_pislquant,
            fov_z=slice_thickness,
            n_readout=n_readout_rx_gain,
        )
        seq.add_block(pp.make_delay(1.0))

        if mrd_header_file:
            for _ in range(ge_pislquant):
                acq = ismrmrd.Acquisition()
                acq.resize(trajectory_dimensions=2, number_of_samples=n_readout_rx_gain)
                prot.append_acquisition(acq)

    t2_prep_echo_times = np.array([0.03, 0.05, 0.1])  # [s]

    # define T1prep settings
    rf_inv_duration = 12e-3  # duration of adiabatic inversion pulse [s]
    rf_inv_spoil_risetime = 0.6e-3  # rise time of spoiler after inversion pulse [s]
    rf_inv_spoil_flattime = 8.4e-3  # flat time of spoiler after inversion pulse [s]
    rf_inv_mu = 4.1  # constant determining amplitude of frequency sweep of adiabatic inversion pulse

    # get prep block duration and calculate corresponding trigger delay
    t1prep_block, t1prep_dur, time_since_inversion = add_t1_inv_prep(
        rf_duration=rf_inv_duration,
        spoiler_ramp_time=rf_inv_spoil_risetime,
        spoiler_flat_time=rf_inv_spoil_flattime,
        rf_mu=rf_inv_mu,
        system=system,
    )

    rot_matrix = _rotation_matrix(delta_array)

    n_repetitions = 1
    repetition_wait_time = 12
    for rep_ in range(n_repetitions):
        rf_max_fa_signal = rf_max_fa.signal.copy()
        for slice_ in range(n_slices):
            # initialize spoke counter
            spoke_counter = 0

            # loop over all blocks
            for block in range(n_blocks):
                if block % 5 == 0:
                    constant_trig_delay = (
                        min_cardiac_trigger_delay - t1prep_dur - ge_segment_delay
                    )  # delay after inversion segment

                    # add trigger and constant part of trigger delay
                    seq.add_block(
                        pp.make_trigger(
                            channel='physio1',
                            duration=round_to_raster(
                                constant_trig_delay - ge_segment_delay, raster_time=system.block_duration_raster
                            ),
                        ),
                        pp.make_label(type='SET', label='TRID', value=1044),
                    )

                    # add all events of T1prep block
                    for idx in t1prep_block.block_events:
                        seq.add_block(t1prep_block.get_block(idx))

                # add no preparation for every block following an inversion block
                elif block % 5 == 1:
                    # add trigger and trigger delay(s)
                    seq.add_block(
                        pp.make_trigger(
                            channel='physio1',
                            duration=round_to_raster(
                                min_cardiac_trigger_delay - ge_segment_delay, raster_time=system.block_duration_raster
                            ),
                        ),
                        pp.make_label(type='SET', label='TRID', value=1048),
                    )

                # add T2prep for every other block
                else:
                    # get prep block duration and calculate corresponding trigger delay
                    echo_idx = block % 5 - 2
                    echo_time = t2_prep_echo_times[echo_idx]
                    t2prep_block, prep_dur = add_t2_prep(echo_time=echo_time, system=system)
                    constant_trig_delay = min_cardiac_trigger_delay - prep_dur

                    # add trigger and constant part of trigger delay
                    seq.add_block(
                        pp.make_trigger(
                            channel='physio1',
                            duration=round_to_raster(
                                constant_trig_delay - ge_segment_delay, raster_time=system.block_duration_raster
                            ),
                        ),
                        pp.make_label(type='SET', label='TRID', value=int(1045 + echo_idx)),
                    )

                    for idx in t2prep_block.block_events:
                        seq.add_block(t2prep_block.get_block(idx))

                # loop over shots / repetitions per block
                for _ in range(n_shots_per_block):
                    # calculate theoretical golden angle rotation for current shot
                    golden_angle = (
                        (rep_ * flip_angles.size + spoke_counter) * 2 * np.pi * (1 - 2 / (1 + np.sqrt(5)))
                    ) % (2 * np.pi)

                    # find closest unique spiral to current golden angle rotation
                    diff = np.abs(delta_array - golden_angle)
                    spiral_idx = np.argmin(diff)

                    # scale rf signal to current flip angle
                    rf_max_fa.signal = rf_max_fa_signal * flip_angles[spoke_counter] / np.max(flip_angles)

                    # add slice selective excitation pulse
                    seq.add_block(
                        rf_max_fa,
                        gz_max_fa,
                        pp.make_label(type='SET', label='TRID', value=1),
                        pp.make_label(label='REP', type='SET', value=rep_),
                        pp.make_label(label='LIN', type='SET', value=spoke_counter),
                        pp.make_label(label='SLC', type='SET', value=slice_),
                    )
                    seq.add_block(gzr_max_fa)

                    if te_delay > 0:
                        seq.add_block(pp.make_delay(te_delay))

                    if use_ext_rot:
                        # Create rotation event
                        rot = pp.make_rotation(rot_matrix[spiral_idx])

                        # Read-prewinding and phase encoding gradients
                        seq.add_block(gx[0], gy[0], adc, rot)
                    else:
                        rotation_angle = delta_array[spiral_idx]
                        seq.add_block(
                            *pp.rotate(gx[0], gy[0], adc, angle=rotation_angle, axis='z', system=system),
                        )

                    seq.add_block(gz_spoil)

                    # add delay in case TR > min_TR
                    if tr_delay > 0:
                        seq.add_block(pp.make_delay(tr_delay))

                    # increment spoke counter
                    spoke_counter += 1

                    if mrd_header_file:
                        # add acquisitions to metadata
                        spiral_trajectory = np.zeros((trajectory.shape[1], 2), dtype=np.float32)

                        # the spiral trajectory is calculated in units of delta_k. for image reconstruction we use delta_k = 1
                        spiral_trajectory[:, 0] = trajectory[spiral_idx, :, 0] * fov_xy
                        spiral_trajectory[:, 1] = trajectory[spiral_idx, :, 1] * fov_xy

                        acq = ismrmrd.Acquisition()
                        acq.resize(trajectory_dimensions=2, number_of_samples=adc.num_samples)
                        acq.traj[:] = spiral_trajectory
                        prot.append_acquisition(acq)

        # add delay between repetitions
        if rep_ < n_repetitions - 1:
            seq.add_block(
                pp.make_delay(
                    round_to_raster(repetition_wait_time - ge_segment_delay, raster_time=system.block_duration_raster)
                ),
                pp.make_label(type='SET', label='TRID', value=2088),
            )

    # obtain noise samples
    seq.add_block(
        pp.make_delay(0.1),
        pp.make_label(label='LIN', type='SET', value=0),
        pp.make_label(label='SLC', type='SET', value=0),
        pp.make_label(type='SET', label='TRID', value=2099),
    )
    seq.add_block(adc, pp.make_label(label='NOISE', type='SET', value=True))
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    if mrd_header_file:
        acq = ismrmrd.Acquisition()
        acq.resize(trajectory_dimensions=2, number_of_samples=adc.num_samples)
        prot.append_acquisition(acq)

    # close ISMRMRD file
    if mrd_header_file:
        prot.close()

    return seq, min_te, min_tr


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    tr: float | None = None,
    rf_flip_angle: float = 12,
    fov_xy: float = 128e-3,
    n_readout: int = 128,
    readout_oversampling: Literal[1, 2, 4] = 2,
    n_spiral_arms: int = 128,
    slice_thickness: float = 8e-3,
    n_slices: int = 1,
    n_dummy_excitations: int = 20,
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
    v141_compatibility: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate a spiral FLASH sequence.

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
    n_readout
        Number of frequency encoding steps.
    readout_oversampling
        Readout oversampling. Determines the number of ADC samples along a spiral and the bandwidth.
    n_spiral_arms
        Number of spiral arms.
    slice_thickness
        Slice thickness of the 2D slice (in meters).
    n_slices
        Number of slices.
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
        Sequence object of spiral FLASH sequence.
    file_path
        Path to the sequence file.
    """
    if system is None:
        system = sys_defaults

    # define settings of rf excitation pulse
    rf_duration = 1.28e-3  # duration of the rf excitation pulse [s]
    rf_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]
    rf_apodization = 0.5  # apodization factor of rf excitation pulse

    # gradient timing
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]

    # define spoiling
    gz_spoil_duration = 0.8e-3  # duration of spoiler gradient [s]
    gz_spoil_area = 4 / slice_thickness  # area / zeroth gradient moment of spoiler gradient
    rf_spoiling_phase_increment = 117  # RF spoiling phase increment [°]. Set to 0 for no RF spoiling.

    # define sequence filename
    filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}nx_{n_spiral_arms}na_{n_slices}ns'
    filename += f'_{readout_oversampling}os'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    seq, min_te, min_tr = spiral_flash_kernel(
        system=system,
        te=te,
        tr=tr,
        fov_xy=fov_xy,
        n_readout=n_readout,
        readout_oversampling=readout_oversampling,
        spiral_undersampling=n_readout / n_spiral_arms,
        slice_thickness=slice_thickness,
        n_slices=n_slices,
        n_dummy_excitations=n_dummy_excitations,
        spiral_sampling_period=None,
        g_spiral_rew_slew_rate_scaling=0.5,
        gx_pre_duration=gx_pre_duration,
        rf_duration=rf_duration,
        rf_flip_angle=rf_flip_angle,
        rf_bwt=rf_bwt,
        rf_apodization=rf_apodization,
        rf_spoiling_phase_increment=rf_spoiling_phase_increment,
        gz_spoil_duration=gz_spoil_duration,
        gz_spoil_area=gz_spoil_area,
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
    seq.set_definition('FOV', [fov_xy, fov_xy, slice_thickness * n_slices])
    seq.set_definition('ReconMatrix', (n_readout, n_readout, 1))
    seq.set_definition('SliceThickness', slice_thickness)
    seq.set_definition('TE', te or min_te)
    seq.set_definition('TR', tr or min_tr)

    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    write_sequence(seq, str(output_path / filename), create_signature=True, v141_compatibility=v141_compatibility)

    if show_plots:
        seq.plot(time_range=(0, 10 * (tr or min_tr)))

    return seq, output_path / filename


if __name__ == '__main__':
    main()
