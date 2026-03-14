"""PROPELLER turbo-spin echo sequence for ADC mapping."""

from collections.abc import Sequence
from pathlib import Path

import ismrmrd
import numpy as np
import pypulseq as pp

from mrseq.preparations.diffusion_prep import DiffusionPrep
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils import write_sequence
from mrseq.utils.constants import GOLDEN_ANGLE_HALF_CIRCLE
from mrseq.utils.ismrmrd import Fov
from mrseq.utils.ismrmrd import Limits
from mrseq.utils.ismrmrd import MatrixSize
from mrseq.utils.ismrmrd import create_header


def adc_tse_propeller_kernel(
    system: pp.Opts,
    te: float | None,
    n_echoes: int,
    tr: float,
    fov_xy: float,
    fov_z: float,
    n_readout: int,
    n_phase_encoding: int,
    phase_encoding_oversampling: float,
    n_slice_encoding: int,
    gx_pre_duration: float,
    gx_flat_time: float,
    rf_ex_duration: float,
    rf_ex_bwt: float,
    rf_ref_duration: float,
    rf_ref_bwt: float,
    rf_ref_width_scale_factor: float,
    readout_oversampling: int,
    gz_crusher_duration: float,
    ge_segment_delay: float,
    b_values: Sequence[float],
    g_diff_max_amplitude: float,
    g_diff_delta_time: float,
    mrd_header_file: str | Path | None,
) -> tuple[pp.Sequence, float]:
    """Generate a PROPELLER turbo-spin echo sequence for ADC mapping..

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
    n_phase_encoding
        Number of phase encoding steps.
    phase_encoding_oversampling
        Oversampling factor for phase encoding to reduce gaps between blades.
    n_slice_encoding
        Number of slice encoding steps.
    gx_pre_duration
        Duration of the readout pre-winder gradient in seconds.
    gx_flat_time
        Flat time of the readout gradient in seconds.
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
    b_values
        b-values for diffusion weighting gradients in s/mm^2
    g_diff_max_amplitude
        Max amplitude of diffusion gradient.
    g_diff_delta_time
        Time between beginning of first and second diffusion gradient.
    ge_segment_delay
        Delay time at the end of each segment for GE scanners.
    mrd_header_file
        Filename of the ISMRMRD header file to be created. If None, no header file is created.

    Returns
    -------
    seq
        PyPulseq Sequence object
    min_te
        Shortest possible echo time.
    """
    if phase_encoding_oversampling < 1:
        raise ValueError('Phase encoding oversampling factor must be larger than or equal to 1.')

    # create PyPulseq Sequence object and set system limits
    seq = pp.Sequence(system=system)

    # create diffusion gradient object
    diff_prep = DiffusionPrep(
        system,
        rf_ref_duration=rf_ref_duration,
        rf_ref_bwt=rf_ref_bwt,
        rf_ref_width_scale_factor=rf_ref_width_scale_factor,
        g_crusher_duration=gz_crusher_duration,
        g_amplitude=g_diff_max_amplitude,
        g_delta_time=g_diff_delta_time,
        max_b_value=max(b_values),
    )

    # create slice selective excitation pulse and gradient
    rf_ex, gz_ex, gzr_ex = pp.make_sinc_pulse(
        flip_angle=90 / 180 * np.pi,
        duration=rf_ex_duration,
        slice_thickness=fov_z,
        apodization=0.5,
        phase_offset=0,
        time_bw_product=rf_ex_bwt,
        delay=system.rf_dead_time,  # delay should equal at least the dead time of the RF pulse
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

    # create frequency encoding pre- and re-winder gradient
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2 - delta_k / 2, duration=gx_pre_duration, system=system)
    gx_post = pp.make_trapezoid(channel='x', area=-gx.area / 2 + delta_k / 2, duration=gx_pre_duration, system=system)
    k0_center_id = np.where((np.arange(n_readout_with_oversampling) - n_readout_with_oversampling / 2) * delta_k == 0)[
        0
    ][0]

    n_blades = int(n_phase_encoding / n_echoes * phase_encoding_oversampling)

    # phase encoding gradient prepared similar to the readout to ensure it can be "mixed" when rotating
    gy_pre_max = pp.make_trapezoid(channel='y', area=delta_k * n_readout / 2, duration=gx_pre_duration, system=system)

    # slice encoding gradient
    gz_pre_max = pp.make_trapezoid(
        channel='z', area=1 / fov_z * n_slice_encoding / 2, duration=gx_pre_duration, system=system
    )

    # calculate minimum delays
    # tau1: between excitation pulse and first refocusing pulse
    min_tau1 = rf_ex.shape_dur / 2
    min_tau1 += max(rf_ex.ringdown_time, gz_ex.fall_time)
    min_tau1 += pp.calc_duration(gzr_ex)
    min_tau1 += diff_prep._time_to_refocusing_pulse
    min_tau1 += max(diff_prep._rf_ref.delay, diff_prep._gz_ref.delay + diff_prep._gz_ref.rise_time)
    min_tau1 += diff_prep._rf_ref.shape_dur / 2

    # tau2: between first refocusing pulse and readout
    min_tau2 = diff_prep._rf_ref.shape_dur / 2
    min_tau2 += max(diff_prep._rf_ref.ringdown_time, diff_prep._gz_ref.fall_time)
    min_tau2 += diff_prep._time_to_refocusing_pulse
    min_tau2 += pp.calc_duration(gx_pre)
    min_tau2 += k0_center_id * adc.dwell
    min_tau2 += adc.dwell / 2
    min_tau2 += max(adc.delay, gx.delay + gx.rise_time)

    # tau3: between readout and next refocusing pulse
    min_tau3 = k0_center_id * adc.dwell
    min_tau3 -= adc.dwell / 2
    min_tau3 += max(gx.fall_time, adc.dead_time)
    min_tau3 += pp.calc_duration(gx_post)
    min_tau3 += pp.calc_duration(diff_prep._g_crush)
    min_tau3 += max(diff_prep._rf_ref.delay, diff_prep._gz_ref.delay + diff_prep._gz_ref.rise_time)
    min_tau3 += diff_prep._rf_ref.shape_dur / 2

    # tau4: between refocusing pulses (no diffusion gradient) and readout
    min_tau4 = diff_prep._rf_ref.shape_dur / 2
    min_tau4 += max(diff_prep._rf_ref.ringdown_time, diff_prep._gz_ref.fall_time)
    min_tau4 += pp.calc_duration(diff_prep._g_crush)
    min_tau4 += pp.calc_duration(gx_pre)
    min_tau4 += k0_center_id * adc.dwell
    min_tau4 += adc.dwell / 2
    min_tau4 += max(adc.delay, gx.delay + gx.rise_time)

    min_te = 2 * max(
        round_to_raster(min_tau1, system.block_duration_raster),
        round_to_raster(min_tau2, system.block_duration_raster),
        round_to_raster(min_tau3, system.block_duration_raster),
        round_to_raster(min_tau4, system.block_duration_raster),
    )

    # calculate echo time delay (te_delay)
    te = min_te if te is None else round_to_raster(te, system.block_duration_raster)
    if te < min_te:
        raise ValueError(f'TE must be larger than {min_te * 1000:.3f} ms. Current value is {te * 1000:.3f} ms.')

    tau1 = round_to_raster(te / 2 - min_tau1, raster_time=system.grad_raster_time)
    tau2 = round_to_raster(te / 2 - min_tau2, raster_time=system.grad_raster_time)
    tau3 = round_to_raster(te / 2 - min_tau3, raster_time=system.grad_raster_time)
    tau4 = round_to_raster(te / 2 - min_tau4, raster_time=system.grad_raster_time)
    print(f'\nCurrent echo time = {(te) * 1000:.3f} ms')

    # create header
    if mrd_header_file:
        hdr = create_header(
            traj_type='other',
            encoding_fov=Fov(x=fov_xy * readout_oversampling, y=fov_xy, z=fov_z),
            recon_fov=Fov(x=fov_xy, y=fov_xy, z=fov_z),
            encoding_matrix=MatrixSize(
                n_x=n_readout_with_oversampling, n_y=n_readout_with_oversampling, n_z=n_slice_encoding
            ),
            recon_matrix=MatrixSize(n_x=n_readout, n_y=n_readout, n_z=n_slice_encoding),
            dwell_time=adc.dwell,
            k1_limits=Limits(min=0, max=n_phase_encoding, center=n_phase_encoding // 2),
            k2_limits=Limits(min=0, max=n_slice_encoding, center=n_slice_encoding // 2),
            h1_resonance_freq=system.gamma * system.B0,
        )

        # write header to file
        prot = ismrmrd.Dataset(mrd_header_file, 'w')
        prot.write_xml_header(hdr.toXML('utf-8'))

    # low-high acquisition within each blade
    if np.mod(n_echoes, 2) == 1:
        pe_idx = np.arange(-(n_echoes - 1) // 2, (n_echoes + 1) // 2)
    else:
        pe_idx = np.arange(-n_echoes // 2, n_echoes // 2)
    pe_idx = pe_idx[np.argsort(np.abs(pe_idx), kind='stable')]

    # add all events to the sequence
    for b_idx, b_value in enumerate(b_values):
        dw_label = pp.make_label(type='SET', label='ECO', value=int(b_idx))
        for se in range(n_slice_encoding):
            se_label = pp.make_label(type='SET', label='PAR', value=int(se))

            # phase encoding along se
            gz_pre = pp.scale_grad(gz_pre_max, (se - n_slice_encoding / 2) / (n_slice_encoding / 2))

            for blade in range(n_blades):
                _start_time_tr_block = sum(seq.block_durations.values())

                # calculate rotation angle for the current spoke
                rotation_angle_rad = GOLDEN_ANGLE_HALF_CIRCLE * (blade + 1)
                # add excitation pulse
                seq.add_block(rf_ex, gz_ex, pp.make_label(type='SET', label='TRID', value=1))
                seq.add_block(gzr_ex)
                seq.add_block(pp.make_delay(tau1))

                seq, _b_value_calc = diff_prep.add_diffusion_prep(seq, b_value=b_value)
                seq.add_block(pp.make_delay(tau2))

                for echo in range(n_echoes):
                    pe_label = pp.make_label(type='SET', label='LIN', value=int(blade * n_echoes + echo))

                    # phase encoding along pe
                    pe_step = pe_idx[echo]
                    gy_pre = pp.scale_grad(gy_pre_max, pe_step / (n_readout / 2))

                    # add pre gradients and all labels
                    labels = [se_label, pe_label, dw_label]
                    seq.add_block(*pp.rotate(gx_pre, gy_pre, angle=rotation_angle_rad, axis='z'), gz_pre)

                    # readout gradient and adc
                    seq.add_block(*pp.rotate(gx, angle=rotation_angle_rad, axis='z'), adc, *labels)

                    # rewind gradients
                    seq.add_block(
                        *pp.rotate(gx_post, pp.scale_grad(gy_pre, -1), angle=rotation_angle_rad, axis='z'),
                        pp.scale_grad(gz_pre, -1),
                    )

                    if echo < n_echoes - 1:
                        seq.add_block(pp.make_delay(tau3))

                        # add refocusing pulse with crusher gradients
                        seq.add_block(diff_prep._g_crush)
                        seq.add_block(diff_prep._rf_ref, diff_prep._gz_ref)
                        seq.add_block(diff_prep._g_crush)

                        seq.add_block(pp.make_delay(tau4))

                    if mrd_header_file:
                        # add acquisitions to metadata
                        k_line = np.linspace(
                            -n_readout_with_oversampling // 2,
                            (n_readout_with_oversampling // 2) - 1,
                            n_readout_with_oversampling,
                        )
                        trajectory = np.zeros((n_readout_with_oversampling, 3), dtype=np.float32)

                        trajectory[:, 0] = k_line * np.cos(rotation_angle_rad) - pe_step * np.sin(rotation_angle_rad)
                        trajectory[:, 1] = k_line * np.sin(rotation_angle_rad) + pe_step * np.cos(rotation_angle_rad)
                        trajectory[:, 2] = se - n_slice_encoding / 2

                        acq = ismrmrd.Acquisition()
                        acq.resize(trajectory_dimensions=3, number_of_samples=adc.num_samples)
                        acq.traj[:] = trajectory
                        prot.append_acquisition(acq)

                duration_tr_block = sum(seq.block_durations.values()) - _start_time_tr_block
                tr_delay = round_to_raster(tr - duration_tr_block - ge_segment_delay, system.block_duration_raster)
                if tr_delay < 0:
                    raise ValueError('Desired TR too short for given sequence parameters.')
                seq.add_block(pp.make_delay(tr_delay))

    # obtain noise samples
    seq.add_block(
        pp.make_label(label='LIN', type='SET', value=0),
        pp.make_label(label='SLC', type='SET', value=0),
        pp.make_label(type='SET', label='TRID', value=9999),
    )
    seq.add_block(adc, pp.make_label(label='NOISE', type='SET', value=True))
    seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
    seq.add_block(pp.make_delay(system.rf_dead_time))

    if mrd_header_file:
        acq = ismrmrd.Acquisition()
        acq.resize(trajectory_dimensions=3, number_of_samples=adc.num_samples)
        prot.append_acquisition(acq)

    # close ISMRMRD file
    if mrd_header_file:
        prot.close()

    return seq, min_te


def main(
    system: pp.Opts | None = None,
    te: float | None = None,
    n_echoes: int = 10,
    tr: float = 4,
    fov_xy: float = 128e-3,
    fov_z: float = 80e-3,
    n_readout: int = 128,
    n_phase_encoding: int = 128,
    phase_encoding_oversampling: float = 1.0,
    n_slice_encoding: int = 10,
    b_values: Sequence[float] = [400, 0, 200, 400, 600],
    show_plots: bool = True,
    test_report: bool = True,
    timing_check: bool = True,
    v141_compatibility: bool = True,
) -> tuple[pp.Sequence, Path]:
    """Generate PROPELLER turbo-spin echo sequence for ADC mapping..

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
    n_phase_encoding
        Number of phase encoding steps.
    phase_encoding_oversampling
        Oversampling factor for phase encoding to reduce gaps between blades.
    n_slice_encoding
        Number of phase encoding steps along the slice direction.
    b_values
        b-values for diffusion weighting gradients in s/mm^2
    show_plots
        Toggles sequence plot.
    test_report
        Toggles advanced test report.
    timing_check
        Toggles timing check of the sequence.
    v141_compatibility
        Save the sequence in pulseq v1.4.1 for backwards compatibility.
    """
    if system is None:
        system = sys_defaults

    if isinstance(b_values, float):
        b_values = (b_values,)

    # define ADC and gradient timing
    readout_oversampling = 2
    adc_dwell = system.grad_raster_time
    gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]
    gx_flat_time = n_readout * adc_dwell  # flat time of readout gradient [s]

    gz_crusher_duration = 1.6e-3  # duration of crusher gradients [s]

    # define settings of rf excitation pulse
    rf_ex_duration = 2e-3  # duration of the rf excitation pulse [s]
    rf_ex_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]

    rf_ref_width_scale_factor = 3.5  # width of refocusing pulse is increased compared to excitation pulse

    # diffusion gradients
    g_diff_delta_time = 40e-3
    g_diff_max_amplitude = system.max_grad * 0.8

    # define sequence filename
    filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_xy_{int(fov_z * 1000)}_fov_z_'
    filename += f'{n_readout}nx_{n_phase_encoding}ny_{n_slice_encoding}nz'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)

    # delete existing header file
    if (output_path / Path(filename + '_header.h5')).exists():
        (output_path / Path(filename + '_header.h5')).unlink()

    seq, min_te = adc_tse_propeller_kernel(
        system=system,
        te=te,
        n_echoes=n_echoes,
        tr=tr,
        fov_xy=fov_xy,
        fov_z=fov_z,
        n_readout=n_readout,
        n_phase_encoding=n_phase_encoding,
        phase_encoding_oversampling=phase_encoding_oversampling,
        n_slice_encoding=n_slice_encoding,
        gx_pre_duration=gx_pre_duration,
        gx_flat_time=gx_flat_time,
        rf_ex_duration=rf_ex_duration,
        rf_ex_bwt=rf_ex_bwt,
        rf_ref_duration=rf_ex_duration * 2,
        rf_ref_bwt=rf_ex_bwt,
        rf_ref_width_scale_factor=rf_ref_width_scale_factor,
        readout_oversampling=readout_oversampling,
        gz_crusher_duration=gz_crusher_duration,
        b_values=b_values,
        g_diff_max_amplitude=g_diff_max_amplitude,
        g_diff_delta_time=g_diff_delta_time,
        ge_segment_delay=0.0,
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
    seq.set_definition('ReadoutOversamplingFactor', readout_oversampling)

    # save seq-file to disk
    output_path = Path.cwd() / 'output'
    output_path.mkdir(parents=True, exist_ok=True)
    # save seq-file to disk
    print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
    write_sequence(seq, str(output_path / filename), create_signature=True, v141_compatibility=v141_compatibility)

    if show_plots:
        seq.plot(time_range=(0, te_list[-1] * 1.2))

    return seq, output_path / filename


if __name__ == '__main__':
    main()
