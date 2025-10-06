"""Tests for sequence helper functions."""

from math import gcd

import numpy as np
import pypulseq as pp
from mrseq.utils import variable_density_spiral_trajectory


def test_multi_gradient_echo(system_defaults):
    """Test multi-echo gradient echo readout as part of a simple sequence."""
    ## BEGIN
    n_readout = 128
    fov_xy = 128e-3
    system = system_defaults
    n_spirals_for_vds_calc = 160
    fov_scaling = [fov_xy, 0]
    oversampling_along_readout = 2
    n_unique_spirals = 16
    adc_dwell_time = 1e-5

    def calc_spiral_catarina(
        system,
        n_readout,
        fov_xy,
        n_spirals_for_vds_calc,
        fov_scaling,
        oversampling_along_readout,
        n_unique_spirals,
        adc_dwell_time,
    ):
        delta_unique_spirals = 2 * np.pi / n_unique_spirals
        delta_array = np.arange(0, 2 * np.pi, delta_unique_spirals)
        delta_array_new = np.array([])
        for i in range(len(delta_array) // 2):
            delta_array_new = np.append(delta_array_new, delta_array[i])

            idx = i + len(delta_array) // 2
            delta_array_new = np.append(delta_array_new, delta_array[idx])

        # calculate spiral trajectory
        r_max = 0.5 / fov_xy * n_readout  # [1/m]
        k, g, s, timing, r, theta = variable_density_spiral_trajectory(
            system=system_defaults,
            sampling_period=system.grad_raster_time,
            n_interleaves=n_spirals_for_vds_calc,
            fov_coefficients=fov_scaling,
            max_kspace_radius=r_max,
        )

        # calculate ADC
        adc_total_samples = np.shape(g)[0] * 2 - 1
        assert adc_total_samples <= 8192, 'ADC samples exceed maximum value of 8192.'
        gc = gcd(n_readout * 2 * oversampling_along_readout, round(system.grad_raster_time / system.adc_raster_time))
        dwell_time_adj = round(adc_dwell_time / system.grad_raster_time * gc) * system.grad_raster_time / gc
        if (dwell_time_adj - adc_dwell_time) / adc_dwell_time > 1e-3:
            print('dwell time adjusted from {dwell_time} to {dwell_time_adj}')
        adc_dwell_time = dwell_time_adj
        adc = pp.make_adc(
            num_samples=adc_total_samples, dwell=adc_dwell_time, system=system, delay=system.adc_dead_time
        )

        # Pre-calculate the spiral gradient waveforms, k-space trajectories, and rewinders
        spiral_readout_grad = np.zeros((n_unique_spirals, 2, np.shape(g)[0]))
        spiral_trajectory = np.zeros((n_unique_spirals, 3, np.shape(k)[0]))
        double_spiral_trajectory = np.zeros((n_unique_spirals, 3, np.shape(k)[0] * 2))
        gx_readout_list = []
        gx_readout_list = []
        gy_readout_list = []
        gx_rewinder_list = []
        gy_rewinder_list = []
        gx_prewinder_list = []
        gy_prewinder_list = []
        double_spiral_readout_x_list = []
        double_spiral_readout_y_list = []

        gx_readout_waveform_all = np.array([])
        readout_duration = 0
        rewinder_duration = 0
        prewinder_duration = 0
        # iterate over all unique spirals
        for n, delta in enumerate(delta_array_new):
            exp_delta = np.exp(1j * delta)
            exp_delta_pi = np.exp(1j * (delta + np.pi))

            spiral_readout_grad[n, 0, :] = np.real(g * exp_delta)
            spiral_readout_grad[n, 1, :] = np.imag(g * exp_delta)
            spiral_trajectory[n, 0, :] = np.real(k * exp_delta_pi)
            spiral_trajectory[n, 1, :] = np.imag(k * exp_delta_pi)

            gx_readout = pp.make_arbitrary_grad(
                channel='x',
                waveform=spiral_readout_grad[n, 0],
                first=0,
                delay=adc.delay,
                system=system,
            )

            gy_readout = pp.make_arbitrary_grad(
                channel='y',
                waveform=spiral_readout_grad[n, 1],
                first=0,
                delay=adc.delay,
                system=system,
            )

            if n % 2 == 0:
                gx_readout.waveform = -np.flip(gx_readout.waveform)
                gy_readout.waveform = -np.flip(gy_readout.waveform)

                gx_readout.area = -gx_readout.area
                gy_readout.area = -gy_readout.area
                gx_new_last = -gx_readout.first
                gy_new_last = -gy_readout.first
                gx_new_first = -gx_readout.last
                gy_new_first = -gy_readout.last
                gx_readout.first = gx_new_first
                gy_readout.first = gy_new_first
                gx_readout.last = gx_new_last
                gy_readout.last = gy_new_last
                spiral_trajectory[n, 0, :] = np.flip(spiral_trajectory[n, 0, :])
                spiral_trajectory[n, 1, :] = np.flip(spiral_trajectory[n, 1, :])

            gx_rewinder, _, _ = pp.make_extended_trapezoid_area(
                area=-gx_readout.area,
                channel='x',
                grad_start=gx_readout.last,
                grad_end=0,
                system=system,
                convert_to_arbitrary=True,
            )

            gy_rewinder, _, _ = pp.make_extended_trapezoid_area(
                area=-gy_readout.area,
                channel='y',
                grad_start=gy_readout.last,
                grad_end=0,
                system=system,
                convert_to_arbitrary=True,
            )

            gx_prewinder, _, _ = pp.make_extended_trapezoid_area(
                area=-gx_readout.area,
                channel='x',
                grad_start=0,
                grad_end=gx_readout.first,
                system=system,
                convert_to_arbitrary=True,
            )

            gy_prewinder, _, _ = pp.make_extended_trapezoid_area(
                area=-gy_readout.area,
                channel='y',
                grad_start=0,
                grad_end=gy_readout.first,
                system=system,
                convert_to_arbitrary=True,
            )

            gx_readout_list.append(gx_readout)
            gy_readout_list.append(gy_readout)
            gx_rewinder_list.append(gx_rewinder)
            gy_rewinder_list.append(gy_rewinder)
            gx_prewinder_list.append(gx_prewinder)
            gy_prewinder_list.append(gy_prewinder)

            # update maximum rewinder duration
            readout_duration = max(readout_duration, max(gx_readout.shape_dur, gy_readout.shape_dur) * 2)
            rewinder_duration = max(rewinder_duration, max(gx_rewinder.shape_dur, gy_rewinder.shape_dur))
            prewinder_duration = max(prewinder_duration, max(gx_prewinder.shape_dur, gy_prewinder.shape_dur))
        gx_rewinder_list = gx_rewinder_list[1::2]
        gy_rewinder_list = gy_rewinder_list[1::2]
        gx_prewinder_list = gx_prewinder_list[::2]
        gy_prewinder_list = gy_prewinder_list[::2]

        prewinder_duration_max = 1e-3
        adc.delay = prewinder_duration_max

        for i in range(len(gx_prewinder_list)):
            gy_prewinder_list[i].delay = prewinder_duration_max - gy_prewinder_list[i].shape_dur
            gx_prewinder_list[i].delay = prewinder_duration_max - gx_prewinder_list[i].shape_dur

        double_spiral_trajectory = double_spiral_trajectory[::2]

        for i in range(len(gx_prewinder_list)):
            double_spiral_trajectory[i, 0, :] = np.append(spiral_trajectory[::2][i, 0], spiral_trajectory[1::2][i, 0])
            gx_double_waveform = np.append(gx_readout_list[::2][i].waveform, gx_readout_list[1::2][i].waveform)
            gx_double_waveform = np.append(gx_prewinder_list[i].waveform, gx_double_waveform)
            gx_double_waveform = np.append(gx_double_waveform, gx_rewinder_list[i].waveform)

            gx_double_readout = pp.make_arbitrary_grad(
                channel='x',
                waveform=gx_double_waveform,
                first=0,
                delay=gx_prewinder_list[i].delay,
                last=0,
                system=system,
            )

            double_spiral_trajectory[i, 1, :] = np.append(spiral_trajectory[::2][i, 1], spiral_trajectory[1::2][i, 1])
            gy_double_waveform = np.append(gy_readout_list[::2][i].waveform, gy_readout_list[1::2][i].waveform)
            gy_double_waveform = np.append(gy_prewinder_list[i].waveform, gy_double_waveform)
            gy_double_waveform = np.append(gy_double_waveform, gy_rewinder_list[i].waveform)

            gy_double_readout = pp.make_arbitrary_grad(
                channel='y',
                waveform=gy_double_waveform,
                first=0,
                delay=gy_prewinder_list[i].delay,
                last=0,
                system=system,
            )

            gx_readout_list[i] = gx_double_readout
            gy_readout_list[i] = gy_double_readout

        gx_readout_list = gx_readout_list[: -n_unique_spirals // 2]
        gy_readout_list = gy_readout_list[: -n_unique_spirals // 2]

        return gx_readout_list, gy_readout_list, adc, double_spiral_trajectory

    def calc_spiral_update(
        system,
        n_readout,
        fov_xy,
        n_spirals_for_vds_calc,
        fov_scaling,
        oversampling_along_readout,
        n_unique_spirals,
        adc_dwell_time,
    ):
        # calculate single spiral trajectory
        traj, grad, s, timing, r, theta = variable_density_spiral_trajectory(
            system=system_defaults,
            sampling_period=system.grad_raster_time,
            n_interleaves=n_spirals_for_vds_calc,
            fov_coefficients=fov_scaling,
            max_kspace_radius=0.5 / fov_xy * n_readout,
        )

        grad = np.concatenate((-np.asarray(grad * np.exp(1j * np.pi))[::-1], grad))
        traj = np.concatenate((np.asarray(traj * np.exp(1j * np.pi))[::-1], traj))
        n_unique_spirals = n_unique_spirals // 2

        # calculate ADC
        adc_total_samples = np.shape(grad)[0]
        if adc_total_samples > 8192:
            raise ValueError(f'Number of ADC samples ({adc_total_samples}) exceed maximum value of 8192.')
        gc = gcd(n_readout * 2 * oversampling_along_readout, round(system.grad_raster_time / system.adc_raster_time))
        dwell_time_adj = round(adc_dwell_time / system.grad_raster_time * gc) * system.grad_raster_time / gc
        if (dwell_time_adj - adc_dwell_time) / adc_dwell_time > 1e-3:
            print('dwell time adjusted from {dwell_time} to {dwell_time_adj}')
        adc_dwell_time = dwell_time_adj
        adc = pp.make_adc(
            num_samples=adc_total_samples, dwell=adc_dwell_time, system=system, delay=system.adc_dead_time
        )

        delta_angle = np.pi / n_unique_spirals

        # Create gradient values and trajectory for different spirals
        grad_x = [np.real(grad * np.exp(1j * delta_angle * idx)) for idx in np.arange(n_unique_spirals)]
        grad_y = [np.imag(grad * np.exp(1j * delta_angle * idx)) for idx in np.arange(n_unique_spirals)]
        traj_x = [np.real(traj * np.exp(1j * delta_angle * idx + np.pi)) for idx in np.arange(n_unique_spirals)]
        traj_y = [np.imag(traj * np.exp(1j * delta_angle * idx + np.pi)) for idx in np.arange(n_unique_spirals)]

        # Create gradient objects
        gx = [pp.make_arbitrary_grad(channel='x', waveform=grad, delay=adc.delay, system=system) for grad in grad_x]
        gy = [pp.make_arbitrary_grad(channel='y', waveform=grad, delay=adc.delay, system=system) for grad in grad_y]

        # Calculate pre- and re-winder gradients
        gx_rew, gx_pre, gy_rew, gy_pre = [], [], [], []
        for gx_, gy_ in zip(gx, gy, strict=True):
            gx_rew.append(
                pp.make_extended_trapezoid_area(
                    area=-gx_.area / 2,
                    channel='x',
                    grad_start=gx_.last,
                    grad_end=0,
                    system=system,
                    convert_to_arbitrary=True,
                )[0]
            )
            gy_rew.append(
                pp.make_extended_trapezoid_area(
                    area=-gy_.area / 2,
                    channel='y',
                    grad_start=gy_.last,
                    grad_end=0,
                    system=system,
                    convert_to_arbitrary=True,
                )[0]
            )

            gx_pre.append(
                pp.make_extended_trapezoid_area(
                    area=-gx_.area / 2,
                    channel='x',
                    grad_start=0,
                    grad_end=gx_.first,
                    system=system,
                    convert_to_arbitrary=True,
                )[0]
            )

            gy_pre.append(
                pp.make_extended_trapezoid_area(
                    area=-gy_.area / 2,
                    channel='y',
                    grad_start=0,
                    grad_end=gy_.first,
                    system=system,
                    convert_to_arbitrary=True,
                )[0]
            )

        prewinder_duration_max = 1e-3
        adc.delay = prewinder_duration_max

        for i in range(len(gx_pre)):
            gy_pre[i].delay = prewinder_duration_max - gy_pre[i].shape_dur
            gx_pre[i].delay = prewinder_duration_max - gx_pre[i].shape_dur

        def combine_gradients(*grad_objects, channel):
            waveform_combined = np.concatenate([grad.waveform for grad in grad_objects])

            return pp.make_arbitrary_grad(
                channel=channel,
                waveform=waveform_combined,
                first=0,
                delay=grad_objects[0].delay,
                last=0,
                system=system,
            )

        gx_combined = [
            combine_gradients(gx_pre, gx_in_out, gx_rew, channel='x')
            for gx_pre, gx_in_out, gx_rew in zip(gx_pre, gx, gx_rew, strict=False)
        ]
        gy_combined = [
            combine_gradients(gy_pre, gy_in_out, gy_rew, channel='y')
            for gy_pre, gy_in_out, gy_rew in zip(gy_pre, gy, gy_rew, strict=False)
        ]
        trajectory = np.stack((np.asarray(traj_x), np.asarray(traj_y), np.zeros_like(traj_y)), axis=-1)

        return gx_combined, gy_combined, adc, trajectory

    import matplotlib.pyplot as plt

    # Compare both waveforms
    def get_interp_waveform(seq, dt=None, scale=1):
        w = seq.waveforms_and_times()
        gx_waveform = w[0][0][1] * scale
        gx_waveform_time = w[0][0][0]
        if dt is None:
            dt = np.arange(gx_waveform_time[0], gx_waveform_time[-1], step=1e-6)
        gx_waveform_intp = np.interp(dt, gx_waveform_time, gx_waveform)

        gy_waveform = w[0][1][1] * scale
        gy_waveform_time = w[0][1][0]
        gy_waveform_intp = np.interp(dt, gy_waveform_time, gy_waveform)

        return gx_waveform_intp, gy_waveform_intp, dt

    gx_readout_list, gy_readout_list, adc, double_spiral_trajectory = calc_spiral_catarina(
        system,
        n_readout,
        fov_xy,
        n_spirals_for_vds_calc,
        fov_scaling,
        oversampling_along_readout,
        n_unique_spirals,
        adc_dwell_time,
    )

    seq = pp.Sequence(system=system_defaults)
    for idx in range(len(gx_readout_list)):
        seq.add_block(gx_readout_list[idx], gy_readout_list[idx], adc)
        seq.add_block(pp.make_delay(1e-3))
    gx_catarina_intp, gy_catarina_intp, dt = get_interp_waveform(seq)

    gx_readout_list, gy_readout_list, adc, double_spiral_trajectory = calc_spiral_update(
        system,
        n_readout,
        fov_xy,
        n_spirals_for_vds_calc,
        fov_scaling,
        oversampling_along_readout,
        n_unique_spirals,
        adc_dwell_time,
    )

    seq = pp.Sequence(system=system_defaults)
    for idx in range(len(gx_readout_list)):
        seq.add_block(gx_readout_list[idx], gy_readout_list[idx], adc)
        seq.add_block(pp.make_delay(1e-3))
    gx_update_intp, gy_update_intp, dt = get_interp_waveform(seq, dt, scale=-1)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(dt, gx_catarina_intp, '-b')
    ax[0].plot(dt, gx_update_intp, ':r')
    ax[0].plot(dt, (gx_update_intp - gx_catarina_intp) * 10, '--k')
    ax[1].plot(dt, gy_catarina_intp, '-b')
    ax[1].plot(dt, gy_update_intp, ':r')
    ax[1].plot(dt, (gy_update_intp - gy_catarina_intp) * 10, '--k')

    # Plot trajectory
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
    plt.figure()
    plt.plot(np.transpose(k_traj))
    plt.figure()
    plt.plot(k_traj[0], k_traj[1], 'b')
    plt.plot(k_traj_adc[0], k_traj_adc[1], 'r.')
    plt.show()

    seq = pp.Sequence(system=system_defaults)
    rf = pp.make_block_pulse(
        flip_angle=np.pi,
        delay=system_defaults.rf_dead_time,
        duration=2e-3,
        phase_offset=0.0,
        system=system_defaults,
    )
    seq.add_block(rf)
    for idx in range(len(gx_readout_list)):
        seq.add_block(gx_readout_list[idx], gy_readout_list[idx], adc)
        seq.add_block(pp.make_delay(3e-3))

    traj = []
    for idx in range(len(gx_readout_list)):
        seq.add_block(gx_readout_list[idx], gy_readout_list[idx], adc)
        seq.add_block(pp.make_delay(3e-3))

        traj.append(double_spiral_trajectory[idx, :, :])

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(traj[0][:, 0], traj[0][:, 1], 'ob-')
    ax[1].plot(traj[0][:, 0], traj[0][:, 1], 'ob-')
    ax[1].plot(traj[1][:, 0], traj[1][:, 1], '+r-')
    for idx in range(len(gx_readout_list)):
        ax[2].plot(traj[idx][:, 0], traj[idx][:, 1], 'ob-')

    ## END

    seq.plot()

    ok, error_report = seq.check_timing()
    if not ok:
        print('\nTiming check failed! Error listing follows\n')
        print(error_report)
    assert ok

    from scipy.signal import argrelextrema

    for spiral_idx in range(len(gx_readout_list)):
        seq = pp.Sequence(system=system_defaults)
        seq.add_block(gx_readout_list[spiral_idx], gy_readout_list[spiral_idx], adc)

        # Get full waveform for readout gradient
        gx_waveform_intp, gy_waveform_intp, dt = get_interp_waveform(seq)
        max_grad = np.max(np.abs(gx_waveform_intp))
        gx_waveform_intp /= max_grad
        gy_waveform_intp /= max_grad

        m0_intp = (np.abs(np.cumsum(gx_waveform_intp)) + np.abs(np.cumsum(gy_waveform_intp))) / (
            2 * len(gx_waveform_intp)
        )
        m0_intp[m0_intp > 1e-3] = 1e-3
        k0_idx = argrelextrema(m0_intp, np.less, order=100)[0]

        # Remove k0-crossings at the beginning and end of the block
        k0_idx = [ki for ki in k0_idx if (ki > 100 and ki < len(dt) - 100)]

        time_of_k0_adc_sample = adc.delay + double_spiral_trajectory.shape[1] // 2 * adc.dwell

        assert len(k0_idx) == 1
        assert m0_intp[0] < 1e-9
        assert m0_intp[-1] < 1e-9
        assert np.isclose(dt[k0_idx], time_to_echo, atol=adc.dwell / 2)
        assert np.isclose(dt[k0_idx], time_of_k0_adc_sample, atol=adc.dwell / 2)

    assert seq is not None

    print(seq.test_report())
