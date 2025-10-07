"""M2D radial FLASH sequence."""

from pathlib import Path

import ismrmrd
import numpy as np
import pypulseq as pp
from pypulseq.rotate import rotate

from mrseq.utils import find_gx_flat_time_on_adc_raster
from mrseq.utils import round_to_raster
from mrseq.utils import sys_defaults
from mrseq.utils.create_ismrmrd_header import create_header

show_plots = True
test_report = True
timing_check = True

te = None
tr = None

fov_xy: float = 128e-3
n_readout: int = 128
n_spokes: int = 128
slice_thickness: float = 8e-3
n_slices: int = 1
receiver_bandwidth_per_pixel: float = 800  # Hz/pixel


system = sys_defaults

# define settings of rf excitation pulse
rf_duration = 1.28e-3  # duration of the rf excitation pulse [s]
rf_flip_angle = 12  # flip angle of rf excitation pulse [°]
rf_bwt = 4  # bandwidth-time product of rf excitation pulse [Hz*s]
rf_apodization = 0.5  # apodization factor of rf excitation pulse
readout_oversampling = 2  # readout oversampling factor, commonly 2. This reduces aliasing artifacts.
spoke_angle = np.pi / 180 * (180 * 0.618034)

# define ADC and gradient timing
n_readout_with_oversampling = int(n_readout * readout_oversampling)
adc_dwell_time = 1.0 / (receiver_bandwidth_per_pixel * n_readout_with_oversampling)
gx_pre_duration = 1.0e-3  # duration of readout pre-winder gradient [s]
gx_flat_time, adc_dwell_time = find_gx_flat_time_on_adc_raster(
    n_readout_with_oversampling, adc_dwell_time, system.grad_raster_time, system.adc_raster_time
)

n_dummy_excitations = 20  # number of dummy excitations before data acquisition to ensure steady state

# define spoiling
gz_spoil_duration = 0.8e-3  # duration of spoiler gradient [s]
gz_spoil_area = 4 / slice_thickness  # area / zeroth gradient moment of spoiler gradient
rf_spoiling_phase_increment = 117  # RF spoiling phase increment [°]. Set to 0 for no RF spoiling.

# define sequence filename
filename = f'{Path(__file__).stem}_{int(fov_xy * 1000)}fov_{n_readout}nx_{n_spokes}na_{n_slices}ns'

output_path = Path.cwd() / 'output'
output_path.mkdir(parents=True, exist_ok=True)

# delete existing header file
if (output_path / Path(filename + '_header.h5')).exists():
    (output_path / Path(filename + '_header.h5')).unlink()

mrd_header_file = (output_path / Path(filename + '_header.h5'),)

# create PyPulseq Sequence object and set system limits
seq = pp.Sequence(system=system)

# create slice selective excitation pulse and gradients
rf, gz, gzr = pp.make_sinc_pulse(  # type: ignore
    flip_angle=rf_flip_angle / 180 * np.pi,
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
delta_k = 1 / fov_xy
gx = pp.make_trapezoid(channel='x', flat_area=n_readout * delta_k, flat_time=gx_flat_time, system=system)
n_readout_with_oversampling = int(n_readout * readout_oversampling)
n_readout_with_oversampling = n_readout_with_oversampling + np.mod(n_readout_with_oversampling, 2)  # make even
adc = pp.make_adc(num_samples=n_readout_with_oversampling, duration=gx.flat_time, delay=gx.rise_time, system=system)

# create frequency encoding pre- and re-winder gradient
gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2 - delta_k / 2, duration=gx_pre_duration, system=system)
gx_post = pp.make_trapezoid(channel='x', area=-gx.area / 2 + delta_k / 2, duration=gx_pre_duration, system=system)
k0_center_id = np.where((np.arange(n_readout_with_oversampling) - n_readout_with_oversampling / 2) * delta_k == 0)[0][0]

# create spoiler gradients
gz_spoil = pp.make_trapezoid(channel='z', system=system, area=gz_spoil_area, duration=gz_spoil_duration)

# calculate minimum echo time
if te is None:
    gzr_gx_dur = pp.calc_duration(gzr, gx_pre)  # gzr and gx_pre are applied simultaneously
else:
    gzr_gx_dur = pp.calc_duration(gzr) + pp.calc_duration(gx_pre)  # gzr and gx_pre are applied sequentially

min_te = (
    rf.shape_dur / 2  # time from center to end of RF pulse
    + max(rf.ringdown_time, gz.fall_time)  # RF ringdown time or gradient fall time
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
    + pp.calc_duration(gz_spoil, gx_post)  # gradient spoiler or readout-re-winder
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
        traj_type='radial',
        fov=fov_xy,
        res=fov_xy / n_readout,
        slice_thickness=slice_thickness,
        dt=adc.dwell,
        n_k1=n_spokes,
    )

    # write header to file
    prot = ismrmrd.Dataset(mrd_header_file, 'w')
    prot.write_xml_header(hdr.toXML('utf-8'))

# obtain noise samples
# seq.add_block(pp.make_label(label='LIN', type='SET', value=0), pp.make_label(label='SLC', type='SET', value=0))
# seq.add_block(adc, pp.make_label(label='NOISE', type='SET', value=True))
# seq.add_block(pp.make_label(label='NOISE', type='SET', value=False))
# seq.add_block(pp.make_delay(system.rf_dead_time))

if mrd_header_file:
    acq = ismrmrd.Acquisition()
    acq.resize(trajectory_dimensions=2, number_of_samples=adc.num_samples)
    prot.append_acquisition(acq)

for slice_ in range(n_slices):
    for spoke_ in range(-n_dummy_excitations, n_spokes):
        # calculate current phase_offset if rf_spoiling is activated
        if rf_spoiling_phase_increment > 0:
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi

        # set frequency offset for current slice
        rf.freq_offset = gz.amplitude * slice_thickness * (slice_ - (n_slices - 1) / 2)

        # add slice selective excitation pulse
        seq.add_block(rf, gz)

        # update rf phase offset for the next excitation pulse
        rf_inc = divmod(rf_inc + rf_spoiling_phase_increment, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        # calculate rotation angle for the current spoke
        rotation_angle_rad = spoke_angle * spoke_

        if te is not None:
            seq.add_block(gzr)
            seq.add_block(pp.make_delay(te_delay))
            seq.add_block(*rotate(gx_pre, angle=rotation_angle_rad, axis='z'))
        else:
            seq.add_block(*rotate(gx_pre, gzr, angle=rotation_angle_rad, axis='z'))

        # rotate and add the readout gradient and ADC
        if spoke_ >= 0:
            labels = []
            labels.append(pp.make_label(label='LIN', type='SET', value=spoke_))
            labels.append(pp.make_label(label='SLC', type='SET', value=slice_))
            seq.add_block(*rotate(gx, adc, angle=rotation_angle_rad, axis='z'), *labels)
        else:
            seq.add_block(pp.make_delay(pp.calc_duration(gx, adc)))

        seq.add_block(*rotate(gx_post, gz_spoil, angle=rotation_angle_rad, axis='z'))

        # add delay in case TR > min_TR
        if tr_delay > 0:
            seq.add_block(pp.make_delay(tr_delay))

        if mrd_header_file and spoke_ >= 0:
            # add acquisitions to metadata
            k_radial_line = np.linspace(
                -n_readout_with_oversampling // 2,
                (n_readout_with_oversampling // 2) - 1,
                n_readout_with_oversampling,
            )
            radial_trajectory = np.zeros((n_readout_with_oversampling, 2), dtype=np.float32)

            radial_trajectory[:, 0] = k_radial_line * np.cos(rotation_angle_rad)
            radial_trajectory[:, 1] = k_radial_line * np.sin(rotation_angle_rad)

            acq = ismrmrd.Acquisition()
            acq.resize(trajectory_dimensions=2, number_of_samples=adc.num_samples)
            acq.traj[:] = radial_trajectory
            prot.append_acquisition(acq)

# close ISMRMRD file
if mrd_header_file:
    prot.close()

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
seq.set_definition('TE', min_te)
seq.set_definition('TR', min_tr)

# save seq-file to disk
print(f"\nSaving sequence file '{filename}.seq' into folder '{output_path}'.")
seq.write(str(output_path / filename), create_signature=True)

if show_plots:
    seq.plot(time_range=(0, 10 * min_tr))
