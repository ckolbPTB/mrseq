# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
import torch
from einops import rearrange
from einops import repeat
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.data import KData
from mrpro.data import KTrajectory
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd

from mrseq.utils.Gmtf import Gmtf
from mrseq.utils.Gmtf import _apply_gmtf_to_gradient_waveform


# %%
def fix_osi_mrd_files(fmrd, fseq, fmrd_new):
    import ismrmrd

    if isinstance(fmrd_new, str):
        fmrd_new = Path(fmrd_new)

    if fmrd_new.exists():
        raise ValueError(f'File {fmrd_new} alsread exists.')

    sequence = pp.Sequence()
    sequence.read(str(fseq))

    adc_labels = sequence.evaluate_labels(evolution='adc')
    # Make labels into lists rather than numpy arrays because ismrmrd cannot deal well with numpy
    for key in adc_labels:
        adc_labels[key] = adc_labels[key].tolist()

    with ismrmrd.File(fmrd, 'r') as file:
        dataset = file[list(file.keys())[-1]]
        ismrmrd_header = dataset.header
        acquisitions = dataset.acquisitions[:]

    if len(adc_labels['AVG']) != len(acquisitions):
        raise ValueError(
            f'Len of AVG labels ({len(adc_labels["AVG"])}) not equal to number of acquisitions ({len(acquisitions)})'
        )

    # Create new file
    ds = ismrmrd.Dataset(fmrd_new)
    ds.write_xml_header(ismrmrd_header.toXML())

    for idx, acq in enumerate(acquisitions):
        acq.idx.average = adc_labels['AVG'][idx]
        ds.append_acquisition(acq)

    ds.close()


# %%
# Scan September 22
frad_raw = (
    '/Users/kolbit01/Documents/Data/GIRF/low_field/2026_09_22/2026-09-22-110203-20260922_105855_radial_with_traj.mrd'
)
frad_seq = '/Users/kolbit01/Documents/Data/GIRF/low_field/2026_09_22/2026-09-22-110203-20260922_105855_radial.seq'

fmrd_orig = '/Users/kolbit01/Documents/Data/GIRF/low_field/2026_09_22/2026-09-22-175828-girf_dyn_triangle_5mm_rt_1cm_avg8_50dwell_200pre_200eddy.mrd'
fgirf_seq = '/Users/kolbit01/Documents/Data/GIRF/low_field/2026_09_22/girf_dyn_triangle_5mm_rt_1cm_avg8_50dwell_200pre_200eddy.seq'
fgirf_raw = '/Users/kolbit01/Documents/Data/GIRF/low_field/2026_09_22/2026-09-22-175828-girf_dyn_triangle_5mm_rt_1cm_avg8_50dwell_200pre_200eddy_mod.mrd'

n_triangles = 5
grad_idx_corr = 0


# Scan September 24
frad_raw = (
    '/Users/kolbit01/Documents/Data/GIRF/low_field/2026_09_24/2026-09-24-121505-20260924_120425_radial_with_traj.mrd'
)
frad_seq = '/Users/kolbit01/Documents/Data/GIRF/low_field/2026_09_22/2026-09-22-110203-20260922_105855_radial.seq'

fmrd_orig = '/Users/kolbit01/Documents/Data/GIRF/low_field/2026_09_24/2026-09-24-080927-girf_dyn_triangle_5mm_rt_1cm_avg8_50dwell_200pre_200eddy.mrd'
fgirf_seq = '/Users/kolbit01/Documents/Data/GIRF/low_field/2026_09_24/girf_dyn_triangle_5mm_rt_1cm_avg8_50dwell_200pre_200eddy.seq'
fgirf_raw = '/Users/kolbit01/Documents/Data/GIRF/low_field/2026_09_24/2026-09-24-080927-girf_dyn_triangle_5mm_rt_1cm_avg8_50dwell_200pre_200eddy_mod.mrd'

n_triangles = 5
grad_idx_corr = 1

# %%
if False:
    fgirf_raw = fmrd_orig.replace('.mrd', '_new.mrd')
    fix_osi_mrd_files(fmrd_orig, fgirf_seq, fgirf_raw)

# %%
gmtf = Gmtf.compute_gmtf(fgirf_raw, fgirf_seq, range(n_triangles))

# %% Visualize GMTF
gmtf.plot(amplitude_lim=(0.3, 5), phase_lim=(-np.pi * 10, np.pi * 10))

# %% Correct GIRF triangles
fig, ax = plt.subplots(3, n_triangles, figsize=(4 * n_triangles, 12))
for triangle_idx in range(n_triangles):
    triangle_waveform = (np.stack((gmtf.grad_time, gmtf.grad_input[triangle_idx])),) * 3
    gradient_waveform_corrected = _apply_gmtf_to_gradient_waveform(triangle_waveform, gmtf.gmtf, gmtf.frequency)

    for grad_idx in range(3):
        line = ax[grad_idx, triangle_idx].plot(triangle_waveform[grad_idx][0], triangle_waveform[grad_idx][1])
        ax[grad_idx, triangle_idx].plot(
            gmtf.grad_time, gmtf.grad_output[grad_idx, triangle_idx, :], color=line[0].get_color(), linestyle='dashed'
        )
        ax[grad_idx, triangle_idx].plot(
            gradient_waveform_corrected[grad_idx][0], gradient_waveform_corrected[grad_idx][1]
        )
        ax[grad_idx, triangle_idx].set_xlim((0.0, 0.002))
        ax[grad_idx, triangle_idx].set_ylim((-0.01, 0.005))


# %% Correction
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
kdata = KData.from_file(frad_raw, KTrajectoryIsmrmrd())
recon = DirectReconstruction(kdata, csm=None)
idata = recon(kdata)
img_uncorr = idata.rss().squeeze().abs().numpy()

ax[2].imshow(img_uncorr[0])
ax[2].set_title('Uncorrected')

gmtf.gmtf = np.stack((gmtf.gmtf[grad_idx_corr, :], gmtf.gmtf[grad_idx_corr, :], gmtf.gmtf[grad_idx_corr, :]))

k_traj_adc = gmtf.correct_gradients(frad_seq)

k_traj_adc[0, :] *= kdata.header.encoding_fov.x
k_traj_adc[1, :] *= kdata.header.encoding_fov.y
k_traj_adc[2, :] *= kdata.header.encoding_fov.z

n_k0 = kdata.data.shape[-1]
k_traj_adc = torch.tensor(k_traj_adc, dtype=torch.float32)
k_traj_reshaped = repeat(k_traj_adc, 'xyz (k1 k0) -> xyz other coils k2 k1 k0', other=1, coils=1, k2=1, k0=n_k0)
k_traj_reshaped = rearrange(k_traj_reshaped, 'xyz 1 coils k2 (k1 other) k0 -> xyz other coils k2 k1 k0', other=14)

k_traj_reshaped[2, ...] = 0.0
ktraj = KTrajectory.from_tensor(
    k_traj_reshaped,
    axes_order='xyz',
    scaling_matrix=None,
    repeat_detection_tolerance=1e-9,
)

ax[0].plot(kdata.traj.kx[0].squeeze(), kdata.traj.ky[0].squeeze(), 'ob')
ax[0].plot(ktraj.kx[0].squeeze(), ktraj.ky[0].squeeze(), '+r')
ax[1].plot(kdata.traj.kx[0].squeeze()[::10, :], kdata.traj.ky[0].squeeze()[::10, :], 'ob')
ax[1].plot(ktraj.kx[0].squeeze()[::10, :], ktraj.ky[0].squeeze()[::10, :], '+r')
ax[1].set_xlim([-10, 10])
ax[1].set_ylim([-10, 10])

kdata.traj = ktraj

recon = DirectReconstruction(kdata, csm=None)
idata = recon(kdata)
img_corr = idata.rss().squeeze().abs().numpy()

ax[3].imshow(img_corr[0])
ax[3].set_title('Corrected')

# %%
