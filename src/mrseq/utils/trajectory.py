"""Basic functionality for trajectory calculation."""

import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp

from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils import variable_density_spiral_trajectory


def cartesian_phase_encoding(
    n_phase_encoding: int,
    acceleration: int = 1,
    n_fully_sampled_center: int = 0,
    sampling_order: Literal['linear', 'low_high', 'high_low', 'random'] = 'linear',
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Cartesian sampling trajectory.

    Parameters
    ----------
    n_phase_encoding
        number of phase encoding points before undersampling
    acceleration
        undersampling factor
    n_fully_sampled_center
        number of phsae encoding points in the fully sampled center. This will reduce the overall undersampling factor.
    sampling_order
        order how phase encoding points are sampled
    """
    if sampling_order == 'random':
        # Linear order of a fully sampled kpe dimension. Undersampling is done later.
        kpe = np.arange(0, n_phase_encoding)
    else:
        # Always include k-space center and more points on the negative side of k-space
        kpe_pos = np.arange(0, n_phase_encoding // 2, acceleration)
        kpe_neg = -np.arange(acceleration, n_phase_encoding // 2 + 1, acceleration)
        kpe = np.concatenate((kpe_neg, kpe_pos), axis=0)

    # Ensure fully sampled center
    kpe_fully_sampled_center = np.arange(
        -n_fully_sampled_center // 2, -n_fully_sampled_center // 2 + n_fully_sampled_center
    )
    kpe = np.unique(np.concatenate((kpe, kpe_fully_sampled_center)))

    # Different temporal orders of phase encoding points
    if sampling_order == 'random':
        perm = np.random.permutation(kpe)
        kpe = kpe[perm[: len(perm) // acceleration]]
    elif sampling_order == 'linear':
        kpe = np.sort(kpe)
    elif sampling_order == 'low_high':
        idx = np.argsort(np.abs(kpe), kind='stable')
        kpe = kpe[idx]
    elif sampling_order == 'high_low':
        idx = np.argsort(-np.abs(kpe), kind='stable')
        kpe = kpe[idx]
    else:
        raise ValueError(f'sampling order {sampling_order} not supported.')

    return kpe, kpe_fully_sampled_center


def undersampled_variable_density_spiral(system: pp.Opts, n_readout: int, fov: float, undersampling_factor: float):
    """Create undersampled variable density spiral.

    The distribution of the k-space points of a spiral trajectory are restricted by the maximum gradient amplitude and
    slew rate. This makes an analytic solution for a given undersampling factor challenging. Here we use an iterative
    approach in order to achieve a variable density spiral with a certain number of readout samplings and undersampling
    factor.

    During the iterative search, the undersampling for the edge of k-space is increased. If this is not enough, then we
    also start to increase the undersampling in the k-space center. The field-of-view varies linearly bewtween the
    k-space center and k-space edge.

    If the undersampling factor is to high, it might not be possible to find a suitable solution.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    n_readout
        Number of readout points per spiral.
    fov
        Field of view (in meters).
    undersampling_factor
        Undersampling factor of spiral trajectory

    Returns
    -------
    tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float)
        - k-space trajectory (traj)
        - Gradient waveform (grad)
        - Slew rate (slew)
        - Time points for the trajectory (timing)
        - Radius values (radius)
        - Angular positions (theta)
        - Number of spiral arms (n_spirals)
        - Scaling of the field-of-view in the k-space center
        - Scaling of the field-of-view in the k-space edge

    """
    # calculate single spiral trajectory
    n_k0 = np.inf
    fov_scaling_center = 1.0
    fov_scaling_edge = 1.0
    n_spirals = int(np.round(n_readout / undersampling_factor))
    while n_k0 > n_readout:
        fov_coefficients = [fov * fov_scaling_center, -fov * (1 - fov_scaling_edge)]

        try:
            traj, grad, slew, timing, radius, theta = variable_density_spiral_trajectory(
                system=system,
                sampling_period=system.grad_raster_time,
                n_interleaves=n_spirals,
                fov_coefficients=fov_coefficients,
                max_kspace_radius=0.5 / fov * n_readout,
            )
            n_k0 = len(grad)
            fov_scaling_edge *= 0.95
        except ValueError:
            # It is not possible to achieve the desired undersampling factor with the given system limits while keeping
            # the full field-of-view in the k-space center. Reduce the field-of-view and try again.
            n_k0 = np.inf
            fov_scaling_center *= 0.95
            fov_scaling_edge = fov_scaling_center

        if fov_scaling_center < 0.1:
            raise ValueError('Cannot find a suitable trajectory.')

    return traj, grad, slew, timing, radius, theta, n_spirals, fov_coefficients[0] / fov, fov_coefficients[1] / fov + 1


def spiral_acquisition(
    system: pp.Opts,
    n_readout: int,
    fov: float,
    undersampling_factor: float,
    readout_oversampling: Literal[1, 2, 4],
    n_spirals: int | None,
    max_pre_duration: float,
    spiral_type: Literal['out', 'in-out'],
):
    """Generate a spiral acquisition sequence.

    Parameters
    ----------
    system
        PyPulseq system limits object.
    n_readout
        Number of readout points per spiral.
    fov
        Field of view (in meters).
    undersampling_factor
        Undersampling factor.
    readout_oversampling
        Oversampling factor for the readout trajectory.
    n_spirals
        Number of spirals to generate. If set to None, this value will be set based on the undersampling factor.
    max_pre_duration : float
        Maximum duration for pre-winder gradients (in seconds).
    spiral_type
        Type of spiral acquisition. 'out' for outward spirals, 'in-out' for spirals turning in and then out.

    Returns
    -------
    gx_combined
        List of combined gradient objects for the x-channel.
    gy_combined
        List of combined gradient objects for the y-channel.
    adc
        PyPulseq ADC object for the acquisition.
    trajectory
        K-space trajectory.
    time_to_echo
        Time to echo from beginning of gradients (in seconds).
    """
    # calculate single spiral trajectory
    traj, grad, _s, _timing, _r, _theta, n_spirals_undersampling, fov_scaling_center, fov_scaling_edge = (
        undersampled_variable_density_spiral(system, n_readout, fov, undersampling_factor)
    )
    n_spirals = n_spirals_undersampling if n_spirals is None else n_spirals
    print(
        f'Target undersampling: {undersampling_factor} - ',
        f'achieved undersampling: {n_readout**2 / (len(traj) * n_spirals_undersampling):.2f}',
        f'FOV: {fov * fov_scaling_center:.3f} (k-sapce center) - {fov * fov_scaling_edge:.3f} (k-space edge)',
    )

    delta_angle = 2 * np.pi / n_spirals
    n_samples_to_echo = 0.5
    if spiral_type == 'in-out':
        n_samples_to_echo = len(grad)
        grad = np.concatenate((-np.asarray(grad * np.exp(1j * np.pi))[::-1], grad))
        traj = np.concatenate((np.asarray(traj * np.exp(1j * np.pi))[::-1], traj))
        delta_angle = delta_angle / 2

    # calculate ADC
    n_readout_with_oversampling = len(grad) * readout_oversampling
    adc_dwell_time = system.grad_raster_time / readout_oversampling
    adc = pp.make_adc(
        num_samples=n_readout_with_oversampling, dwell=adc_dwell_time, system=system, delay=system.adc_dead_time
    )
    traj = np.interp(
        np.linspace(0.5 / readout_oversampling, len(grad) - 0.5 / readout_oversampling, n_readout_with_oversampling),
        np.linspace(0.5, len(grad) - 0.5, len(grad)),
        traj,
    )

    print(f'Receiver bandwidth: {int(1.0 / (adc_dwell_time * n_readout_with_oversampling))} Hz/pixel')

    # Create gradient values and trajectory for different spirals
    grad = [grad * np.exp(1j * delta_angle * idx) for idx in np.arange(n_spirals)]
    traj = [traj * np.exp(1j * delta_angle * idx) for idx in np.arange(n_spirals)]

    # Create gradient objects
    gx = [pp.make_arbitrary_grad(channel='x', waveform=g.real, delay=adc.delay, system=system) for g in grad]
    gy = [pp.make_arbitrary_grad(channel='y', waveform=g.imag, delay=adc.delay, system=system) for g in grad]

    # Calculate pre- and re-winder gradients
    gx_rew, gx_pre, gy_rew, gy_pre = [], [], [], []
    for gx_, gy_ in zip(gx, gy, strict=True):
        gx_rew.append(
            pp.make_extended_trapezoid_area(
                area=-gx_.area if spiral_type == 'out' else -gx_.area / 2,
                channel='x',
                grad_start=gx_.last,
                grad_end=0,
                system=system,
                convert_to_arbitrary=True,
            )[0]
        )
        gy_rew.append(
            pp.make_extended_trapezoid_area(
                area=-gy_.area if spiral_type == 'out' else -gy_.area / 2,
                channel='y',
                grad_start=gy_.last,
                grad_end=0,
                system=system,
                convert_to_arbitrary=True,
            )[0]
        )

        if spiral_type == 'in-out':
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
        else:
            gx_pre.append(None)
            gy_pre.append(None)

    if spiral_type == 'in-out':
        adc.delay = max_pre_duration

        for i in range(len(gx_pre)):
            gy_pre[i].delay = max_pre_duration - gy_pre[i].shape_dur
            gx_pre[i].delay = max_pre_duration - gx_pre[i].shape_dur
    else:
        max_pre_duration = 0.0

    def combine_gradients(*grad_objects, channel):
        grad_list = [grad for grad in grad_objects if grad is not None]  # Remove None
        waveform_combined = np.concatenate([grad.waveform for grad in grad_list])

        return pp.make_arbitrary_grad(
            channel=channel,
            waveform=waveform_combined,
            first=0,
            delay=grad_list[0].delay,
            last=0,
            system=system,
        )

    gx_combined = [
        combine_gradients(gx_pre, gx_in_out, gx_rew, channel='x')
        for gx_pre, gx_in_out, gx_rew in zip(gx_pre, gx, gx_rew, strict=True)
    ]
    gy_combined = [
        combine_gradients(gy_pre, gy_in_out, gy_rew, channel='y')
        for gy_pre, gy_in_out, gy_rew in zip(gy_pre, gy, gy_rew, strict=True)
    ]

    # times -1 to match pulseq trajectory calculation
    trajectory = -np.stack((np.asarray(traj).real, np.asarray(traj).imag), axis=-1)

    time_to_echo = max_pre_duration + n_samples_to_echo * readout_oversampling * adc.dwell

    return gx_combined, gy_combined, adc, trajectory, time_to_echo


class EpiReadout:
    """EPI readout module.

    Attributes
    ----------
    system
        System limits.
    fov
        Field of view in meters (square).
    n_readout
        Number of readout points.
    n_phase_encoding
        Number of phase encoding points.
    bandwidth
        Total receiver bandwidth in Hz (approx. 1/dwell_time_nyquist).
        readout_duration will be approx matrix_size[0] / bandwidth.
    oversampling
        ADC oversampling factor.
    ramp_sampling
        If True, ADC is active during gradient ramps (optimized timing).
    readout_type
        Readout type ('flyback' or 'symmetric').
    partial_fourier_factor
        Partial Fourier factor (0.5 to 1.0).
    adc_freq_offset
        Frequency offset for the ADC.
    pe_enable
        Enable phase encoding (useful for calibration if False).
    spoiling_enable
        Enable spoiling gradients (useful for calibration if False).
    adc
        ADC event.
    gx_pre
        Pre-phaser gradient in readout direction.
    gy_pre
        Pre-phaser gradient in phase encoding direction.
    gx
        Readout gradient.
    gy
        Phase encoding gradient.
    gy_blip
        Complete blip gradient used for flyback readout.
    gy_blipup
        Ramp-up part of blip gradient used for symmetric readout.
    gy_blipdown
        Ramp-down part of blip gradient used for symmetric readout.
    gy_blipdownup
        Combined gy_blipdown and gy_blipup gradient used for symmetric readout.
    gz_spoil
        Spoiler gradient in z-direction.
    gx_flyback
        Flyback gradient in x-direction.
    """

    def __init__(
        self,
        system: pp.Opts | None = None,
        fov: float = 128e-3,
        n_readout: int = 64,
        n_phase_encoding: int = 64,
        bandwidth: float = 50e3,
        oversampling: int = 2,
        ramp_sampling: bool = True,
        readout_type: Literal['flyback', 'symmetric'] = 'symmetric',
        partial_fourier_factor: float = 1.0,
        adc_freq_offset: float = 0,
        pe_enable: bool = True,
        spoiling_enable: bool = True,
    ):
        """Initialize EPI Readout.

        Parameters
        ----------
        system
            System limits.
        fov
            Field of view in meters (square).
        n_readout
            Number of readout points.
        n_phase_encoding
            Number of phase encoding points.
        bandwidth
            Total receiver bandwidth in Hz (approx. 1/dwell_time_nyquist).
            readout_duration will be approx matrix_size[0] / bandwidth.
        oversampling
            ADC oversampling factor.
        ramp_sampling
            If True, ADC is active during gradient ramps (optimized timing).
        readout_type
            Readout type ('flyback' or 'symmetric').
        partial_fourier_factor
            Partial Fourier factor (0.5 to 1.0].
        adc_freq_offset
            Frequency offset for the ADC.
        pe_enable
            Enable phase encoding (useful for calibration if False).
        spoiling_enable
            Enable spoiling gradients (useful for calibration if False).
        """
        if not 0.5 < partial_fourier_factor <= 1.0:
            raise ValueError('Desired partial Fourier factor must be larger than 0.5 and smaller or equal to 1.0.')

        # set system to default if not provided
        if system is None:
            system = sys_defaults

        self.system = system
        self.fov = fov
        self.n_readout = n_readout
        self.n_phase_encoding = n_phase_encoding
        self.bandwidth = bandwidth
        self.oversampling = oversampling
        self.ramp_sampling = ramp_sampling
        self.readout_type = readout_type
        self.partial_fourier_factor = partial_fourier_factor
        self.adc_freq_offset = adc_freq_offset
        self.pe_enable = pe_enable
        self.spoiling_enable = spoiling_enable

        # Derived parameters
        self.readout_time = self.n_readout / self.bandwidth
        self.delta_k = 1 / self.fov

        # Initiate all optional gradients as None
        self.gx_flyback = None
        self.gy_blipdown = None
        self.gy_blipup = None
        self.gy_blipdownup = None
        self.gz_spoil = None

        # Create blip gradient with shortest possible timing
        gy_blip_duration = np.ceil(2 * np.sqrt(self.delta_k / system.max_slew) / 10e-6 / 2) * 10e-6 * 2
        gy_blip_half_dur = gy_blip_duration / 2
        self.gy_blip = pp.make_trapezoid(channel='y', system=self.system, area=-self.delta_k, duration=gy_blip_duration)

        # Create readout gradient
        gx_encoding_area = self.n_readout * self.delta_k
        if self.ramp_sampling:
            # Calculate additional gradient area from gy_blip assuming maximum slew rate
            gy_blip_area = np.power(gy_blip_half_dur, 2) * self.system.max_slew

            # Create gradient with additional area
            gx = pp.make_trapezoid(
                channel='x',
                system=self.system,
                area=gx_encoding_area + gy_blip_area,
                duration=gy_blip_half_dur + self.readout_time + gy_blip_half_dur,
            )

            if not gx.fall_time == gx.rise_time:
                raise ValueError('Gradient fall time must be equal to rise time for ramp sampling.')

            # Second, correct area taking actual slew rate into account
            gx_slew = gx.amplitude / gx.rise_time
            gx_area_reduced_slew = gx.area - np.power(gy_blip_half_dur, 2) * gx_slew

            gx.amplitude = float(gx.amplitude * gx_encoding_area / gx_area_reduced_slew)
            gx.area = float(gx.amplitude * (gx.rise_time / 2 + gx.flat_time + gx.fall_time / 2))
            gx.flat_area = float(gx.amplitude * gx.flat_time)
            self.gx = gx
        else:
            self.gx = pp.make_trapezoid(
                channel='x',
                system=self.system,
                flat_area=gx_encoding_area,
                flat_time=self.readout_time,
            )

        # Create ADC event
        adc_dwell = self.delta_k / self.gx.amplitude / self.oversampling
        adc_dwell = round_to_raster(adc_dwell, self.system.adc_raster_time, 'ceil')
        adc_samples = int(np.ceil(self.readout_time / adc_dwell / 4) * 4)

        adc_time_to_center = adc_dwell * ((adc_samples - 1) / 2 + 0.5)
        adc_delay = self.gx.rise_time + self.gx.flat_time / 2 - adc_time_to_center
        adc_delay = round_to_raster(adc_delay, self.system.adc_raster_time)

        self.adc = pp.make_adc(
            num_samples=adc_samples,
            dwell=adc_dwell,
            freq_offset=self.adc_freq_offset,
            delay=adc_delay,
            system=self.system,
        )

        # Create and align pre-phaser gradients considering partial fourier factor
        # determine the number of pe lines after (and including) k-space center, which is independent of partial fourier
        self.n_phase_enc_post_center = int(np.ceil(self.n_phase_encoding / 2 + 1))
        # find the closest number of lines to the desired partial fourier factor
        valid_n_phase_total = int(np.ceil(self.partial_fourier_factor * self.n_phase_encoding))
        # ensure that at least one line before the center is acquired
        self.n_phase_enc_pre_center = max(1, valid_n_phase_total - self.n_phase_enc_post_center)
        self.n_phase_enc_total = self.n_phase_enc_pre_center + self.n_phase_enc_post_center
        # recalculate the actual partial fourier factor
        actual_pf_factor = self.n_phase_enc_total / self.n_phase_encoding
        if actual_pf_factor != self.partial_fourier_factor:
            warnings.warn(
                f'Desired partial Fourier factor {self.partial_fourier_factor} adjusted to {actual_pf_factor}.',
                UserWarning,
                stacklevel=2,
            )
            self.partial_fourier_factor = actual_pf_factor

        self.gx_pre = pp.make_trapezoid(channel='x', system=self.system, area=-self.gx.area / 2 - self.delta_k / 2)
        self.gy_pre = pp.make_trapezoid(
            channel='y', system=self.system, area=self.n_phase_enc_pre_center * self.delta_k
        )
        self.gx_pre, self.gy_pre = pp.align(right=[self.gx_pre, self.gy_pre])

        if self.readout_type == 'flyback':
            # Create flyback gradient in case of flyback readout
            self.gx_flyback = pp.make_trapezoid(channel='x', system=self.system, area=-self.gx.area)
        elif self.readout_type == 'symmetric':
            # Split and align blip gradient in case of symmetric readout
            gy_parts = pp.split_gradient_at(grad=self.gy_blip, time_point=gy_blip_duration / 2, system=self.system)
            self.gy_blipup, self.gy_blipdown, _ = pp.align(right=gy_parts[0], left=[gy_parts[1], self.gx])
            self.gy_blipdownup = pp.add_gradients((self.gy_blipdown, self.gy_blipup), system=self.system)
        else:
            raise NotImplementedError('Currently, only "symmetric" and "flyback" readout types are supported.')

        # Disable phase encoding if self.pe_enable is False
        if not self.pe_enable:
            self.gy_pre.amplitude = 0
            if (
                self.readout_type == 'symmetric'
                and self.gy_blipup is not None
                and self.gy_blipdown is not None
                and self.gy_blipdownup is not None
            ):
                self.gy_blipup.waveform *= 0
                self.gy_blipdown.waveform *= 0
                self.gy_blipdownup.waveform *= 0

        # Create spoiler gradient if spoiling is enabled
        if self.spoiling_enable:
            self.gz_spoil = pp.make_trapezoid(channel='z', system=self.system, area=4 * self.delta_k * n_readout)

    def add_to_seq(self, seq: pp.Sequence):
        """Add EPI readout blocks to the sequence."""
        # (Re)set phase encoding (LIN) label
        lin_label = pp.make_label(label='LIN', type='SET', value=0)

        # Add pre-phaser gradients
        seq.add_block(self.gx_pre, self.gy_pre)

        for pe_idx in range(self.n_phase_enc_total):
            rev_label = pp.make_label(type='SET', label='REV', value=self.gx.amplitude < 0)
            if self.readout_type == 'symmetric':
                # Select blip gradient based on phase encoding index
                if pe_idx == 0:
                    gy_blip = self.gy_blipup
                elif pe_idx == self.n_phase_enc_total - 1:
                    gy_blip = self.gy_blipdown
                else:
                    gy_blip = self.gy_blipdownup

                # Add readout block and reverse polarity of readout gradient
                seq.add_block(self.gx, gy_blip, self.adc, lin_label, rev_label)
                self.gx.amplitude = -self.gx.amplitude
            elif self.readout_type == 'flyback':
                seq.add_block(self.gx, self.adc, lin_label, rev_label)
                if pe_idx != self.n_phase_enc_total - 1:
                    seq.add_block(self.gx_flyback, self.gy_blip)

            lin_label = pp.make_label(label='LIN', type='INC', value=1)

        if self.spoiling_enable:
            seq.add_block(self.gz_spoil)

        return seq

    def plot_sequence(self):
        """Plot the sequence."""
        seq = pp.Sequence(self.system)
        seq = self.add_to_seq(seq)
        seq.plot(grad_disp='mT/m')

    def plot_trajectory(self):
        """Plot k-space trajectory."""
        seq = pp.Sequence(self.system)
        # add dummy excitation pulse for trajectory calculation
        seq.add_block(
            pp.make_block_pulse(
                flip_angle=np.pi / 2,
                duration=2e-3,
                delay=self.system.rf_dead_time,
                use='excitation',
                system=self.system,
            )
        )
        seq = self.add_to_seq(seq)

        # calculate k-space trajectory
        k_traj_adc, k_traj, _, _, _ = seq.calculate_kspace()

        # plot trajectory
        fig = plt.figure()
        plt.plot(k_traj[0], k_traj[1], 'b')
        plt.plot(k_traj_adc[0], k_traj_adc[1], 'x', color='red', markersize=4)
        plt.grid()
        plt.show()

        return fig
