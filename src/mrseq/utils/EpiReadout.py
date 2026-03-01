"""Echo Planar Imaging (EPI) readout sequence."""

from typing import Literal

import ismrmrd
import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp

from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults


class EpiReadout:
    """EPI readout module supporting flyback and symmetric readout, ramp sampling, oversampling, and partial Fourier.

    Attributes
    ----------
    system
        System limits.
    ramp_sampling
        If True, ADC is active during gradient ramps (optimized timing).
    readout_type
        Readout type ('flyback' or 'symmetric').
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
            Total receiver bandwidth in Hz.
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
            Enable phase encoding (useful for calibration scans if False).
        spoiling_enable
            Enable spoiling gradients.
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
        readout_time = n_readout / bandwidth
        delta_kx = 1 / (fov * oversampling)
        delta_ky = 1 / fov

        # Initiate all optional gradients as None
        self.gx_flyback = None
        self.gy_blipdown = None
        self.gy_blipup = None
        self.gy_blipdownup = None
        self.gz_spoil = None

        # Create blip gradient with shortest possible timing
        gy_blip_duration = np.ceil(2 * np.sqrt(delta_ky / system.max_slew) / 10e-6 / 2) * 10e-6 * 2
        gy_blip_half_dur = gy_blip_duration / 2
        self.gy_blip = pp.make_trapezoid(channel='y', system=self.system, area=-delta_ky, duration=gy_blip_duration)

        # Create readout gradient
        gx_encoding_area = n_readout * delta_kx * oversampling
        if self.ramp_sampling:
            # Calculate additional gradient area from gy_blip assuming maximum slew rate
            extra_area = np.power(gy_blip_half_dur, 2) * self.system.max_slew

            # Create gradient with additional area
            gx = pp.make_trapezoid(
                channel='x',
                system=self.system,
                area=gx_encoding_area + extra_area,
                duration=gy_blip_half_dur + readout_time + gy_blip_half_dur,
            )

            if not gx.fall_time == gx.rise_time:
                raise ValueError('Gradient fall time must be equal to rise time for ramp sampling.')

            # Second, correct area taking actual slew rate into account
            gx_slew = gx.amplitude / gx.rise_time
            gx_area_reduced_slew = gx.area - gx_slew * np.power(gy_blip_half_dur, 2)

            gx.amplitude = float(gx.amplitude * gx_encoding_area / gx_area_reduced_slew)
            gx.area = float(gx.amplitude * (gx.rise_time / 2 + gx.flat_time + gx.fall_time / 2))
            gx.flat_area = float(gx.amplitude * gx.flat_time)
            self.gx = gx
        else:
            self.gx = pp.make_trapezoid(
                channel='x',
                system=self.system,
                flat_area=gx_encoding_area,
                flat_time=readout_time,
            )

        # Create ADC event
        adc_dwell = delta_kx / self.gx.amplitude
        adc_dwell = round_to_raster(adc_dwell, self.system.adc_raster_time)

        adc_samples = int(round(readout_time / adc_dwell / 4) * 4)

        adc_time_to_center = adc_dwell * (adc_samples / 2 + 0.5)
        adc_delay = self.gx.rise_time + self.gx.flat_time / 2 - adc_time_to_center + adc_dwell / 2
        # the adc delay has to be rounded to the rf raster time (not the adc raster time)
        adc_delay = round_to_raster(adc_delay, self.system.rf_raster_time)

        self.adc = pp.make_adc(
            num_samples=adc_samples,
            dwell=adc_dwell,
            freq_offset=adc_freq_offset,
            delay=adc_delay,
            system=self.system,
        )

        # Create and align pre-phaser gradients considering partial fourier factor
        # determine the number of "PE" lines after (and including) k-space center (independent of partial fourier)
        self.n_phase_enc_post_center = int(np.ceil(self.n_phase_encoding / 2 + 1))
        # find the closest number of lines to the desired partial fourier factor
        valid_n_phase_total = int(np.ceil(partial_fourier_factor * self.n_phase_encoding))
        # ensure that at least one line before the center is acquired
        self.n_phase_enc_pre_center = max(1, valid_n_phase_total - self.n_phase_enc_post_center)
        # update the total number of "PE" lines
        self.n_phase_enc_total = self.n_phase_enc_pre_center + self.n_phase_enc_post_center
        # recalculate the actual partial fourier factor
        actual_pf_factor = self.n_phase_enc_total / self.n_phase_encoding
        if actual_pf_factor != partial_fourier_factor:
            print(f'Adjusted partial Fourier factor from {partial_fourier_factor} to {actual_pf_factor:.2f}.')
            self.partial_fourier_factor = actual_pf_factor

        # Create pre-phaser gradients
        self.gx_pre = pp.make_trapezoid(
            channel='x',
            system=self.system,
            area=-self.gx.area / 2 - delta_kx / 2,
        )
        self.gy_pre = pp.make_trapezoid(
            channel='y',
            system=self.system,
            area=self.n_phase_enc_pre_center * delta_ky,
        )
        self.gx_pre, self.gy_pre = pp.align(right=[self.gx_pre, self.gy_pre])

        # Create and align "phase encoding" gradients
        if self.readout_type == 'flyback':
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
            elif self.readout_type == 'flyback':
                self.gy_blip.waveform *= 0

        # Create spoiler gradient if spoiling is enabled
        if self.spoiling_enable:
            self.gz_spoil = pp.make_trapezoid(channel='z', system=self.system, area=4 * delta_ky * n_readout)

    @property
    def time_to_center(self) -> float:
        """Return time from beginning of readout to center of k-space (needed for TE calculations)."""
        # i) add time for pre phaser
        time_to_center = pp.calc_duration(self.gx_pre, self.gy_pre)
        # ii) add time for completed k-space lines before central (ky = 0) line
        if self.readout_type == 'flyback':
            time_to_center += self.n_phase_enc_pre_center * (
                pp.calc_duration(self.gx) + pp.calc_duration(self.gx_flyback, self.gy_blip)
            )
        elif self.readout_type == 'symmetric':
            time_to_center += self.n_phase_enc_pre_center * pp.calc_duration(self.gx)
        # iii) add time before start of ADC of central k-space line
        if self.ramp_sampling:
            time_to_center += self.adc.delay
        else:
            time_to_center += self.gx.rise_time
        # iv) add time from start of ADC (for ky = 0) to timepoint when kx = 0 as well
        time_to_center += self.adc.dwell * (self.adc.num_samples / 2 + 0.5)

        return float(time_to_center)

    @property
    def time_to_center_without_prephaser(self) -> float:
        """Return time from after pre-phasers to center of k-space (needed for TE calculations)."""
        return self.time_to_center - pp.calc_duration(self.gx_pre, self.gy_pre)

    @property
    def total_duration(self) -> float:
        """Return total duration of readout including pre-phaser and optional spoiler."""
        total_duration = pp.calc_duration(self.gx_pre, self.gy_pre)
        if self.readout_type == 'flyback':
            total_duration += self.n_phase_enc_total * (
                pp.calc_duration(self.gx) + pp.calc_duration(self.gx_flyback, self.gy_blip)
            )
            total_duration -= pp.calc_duration(self.gx_flyback, self.gy_blip)
        elif self.readout_type == 'symmetric':
            total_duration += self.n_phase_enc_total * pp.calc_duration(self.gx)
        # add time for spoiler if enabled
        if self.spoiling_enable:
            total_duration += pp.calc_duration(self.gz_spoil)

        return float(total_duration)

    @property
    def total_duration_without_prephaser(self) -> float:
        """Return total duration of readout excluding pre-phasers but including optional spoiler."""
        return self.total_duration - pp.calc_duration(self.gx_pre, self.gy_pre)

    def add_to_seq(
        self,
        seq: pp.Sequence,
        add_prephaser: bool = True,
        mrd_dataset: ismrmrd.Dataset | None = None,
    ) -> tuple[pp.Sequence, ismrmrd.Dataset | None]:
        """Add EPI readout blocks to the sequence."""
        # (Re)set phase encoding (LIN) label
        lin_label = pp.make_label(label='LIN', type='SET', value=0)

        # Add pre-phaser gradients if enabled
        if add_prephaser:
            seq.add_block(self.gx_pre, self.gy_pre)

        for pe_idx in range(self.n_phase_enc_total):
            rev_label = pp.make_label(type='SET', label='REV', value=self.gx.amplitude < 0)
            seg_label = pp.make_label(type='SET', label='SEG', value=self.gx.amplitude < 0)
            if self.readout_type == 'symmetric':
                # Select blip gradient based on phase encoding index
                if pe_idx == 0:
                    gy_blip = self.gy_blipup
                elif pe_idx == self.n_phase_enc_total - 1:
                    gy_blip = self.gy_blipdown
                else:
                    gy_blip = self.gy_blipdownup

                # Add readout block and reverse polarity of readout gradient
                seq.add_block(self.gx, gy_blip, self.adc, lin_label, rev_label, seg_label)
                self.gx.amplitude = -self.gx.amplitude

            elif self.readout_type == 'flyback':
                seq.add_block(self.gx, self.adc, lin_label, rev_label)
                if pe_idx != self.n_phase_enc_total - 1:
                    seq.add_block(self.gx_flyback, self.gy_blip)

            lin_label = pp.make_label(label='LIN', type='INC', value=1)

        if self.spoiling_enable:
            seq.add_block(self.gz_spoil)

        return seq, mrd_dataset

    def plot_sequence(self):
        """Plot the sequence."""
        seq = pp.Sequence(self.system)
        seq, _ = self.add_to_seq(seq)
        _, axs1, _, axs2 = seq.plot(grad_disp='mT/m', plot_now=False)
        # add vertical line at time to center to all plots in both figures
        for ax in axs1 + axs2:
            ax.axvline(x=self.time_to_center, color='r', linestyle='--')
        plt.show()

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
        seq, _ = self.add_to_seq(seq)

        # calculate k-space trajectory
        k_traj_adc, k_traj, _, _, _ = seq.calculate_kspace()

        # plot trajectory
        fig = plt.figure()
        plt.plot(k_traj[0], k_traj[1], 'b')
        plt.plot(k_traj_adc[0], k_traj_adc[1], 'x', color='red', markersize=4)
        plt.grid()
        plt.show()

        return fig
