"""Utilities to deal with creating ISMRMRD files."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ismrmrd
import numpy as np
import pypulseq as pp
from typing_extensions import Self

T_traj = Literal['cartesian', 'epi', 'radial', 'spiral', 'other']


@dataclass(slots=True)
class Limits:
    """Limits dataclass with min, max, and center attributes."""

    min: int = 0
    max: int = 0
    center: int = 0

    @classmethod
    def from_label_list(cls, label_list) -> Self:
        """Create Limits from list of labels."""
        return cls(min=min(label_list), max=max(label_list), center=((max(label_list) - min(label_list) + 1) // 2))


@dataclass(slots=True)
class Fov:
    """Fov (x, y, z)."""

    x: float
    y: float
    z: float


@dataclass(slots=True)
class MatrixSize:
    """Matrix size (x, y, z)."""

    n_x: int
    n_y: int
    n_z: int


def m_to_mm(value: float) -> float:
    """Convert meters to millimeters."""
    return value * 1e3


def create_header(
    traj_type: T_traj,
    encoding_fov: Fov,
    recon_fov: Fov,
    encoding_matrix: MatrixSize,
    recon_matrix: MatrixSize,
    dwell_time: float,
    k1_limits: Limits | None = None,
    k2_limits: Limits | None = None,
    slice_limits: Limits | None = None,
    contrast_limits: Limits | None = None,
    average_limits: Limits | None = None,
    repetition_limits: Limits | None = None,
    phase_limits: Limits | None = None,
    set_limits: Limits | None = None,
    h1_resonance_freq: float = 127729200,  # 3T
) -> ismrmrd.xsd.ismrmrdHeader:
    """
    Create an ISMRMRD header based on the given parameters.

    This code is adjusted from https://github.com/mrphysics-bonn/spiral-pypulseq-example

    Parameters
    ----------
    traj_type
        Trajectory type.
    encoding_fov
        Field of view encoded by the gradients in meters.
    recon_fov
        Field of view for reconstruction (e.g. without readout oversampling) in meters.
    encoding_matrix
        Matrix size of encoded k-space.
    recon_matrix
        Matrix size for reconstruction.
    dwell_time
        Dwell time in seconds.
    k1_limits
        Min, max, and center limits for k1.
    k2_limits
        Min, max, and center limits for k2.
    slice_limits
        Min, max, and center limits for slices.
    contrast_limits
        Min, max, and center limits for contrast.
    average_limits
        Min, max, and center limits for averages.
    repetition_limits
        Min, max, and center limits for repetitions.
    phase_limits
        Min, max, and center limits for phases.
    set_limits
        Min, max, and center limits for set.
    h1_resonance_freq
        Resonance frequency of water nuclei.

    Returns
    -------
        created ISMRMRD header.
    """
    if k1_limits is None:
        k1_limits = Limits()
    if k2_limits is None:
        k2_limits = Limits()
    if slice_limits is None:
        slice_limits = Limits()
    if contrast_limits is None:
        contrast_limits = Limits()
    if average_limits is None:
        average_limits = Limits()
    if repetition_limits is None:
        repetition_limits = Limits()
    if phase_limits is None:
        phase_limits = Limits()
    if set_limits is None:
        set_limits = Limits()

    hdr = ismrmrd.xsd.ismrmrdHeader()

    # experimental conditions
    exp = ismrmrd.xsd.experimentalConditionsType()
    exp.H1resonanceFrequency_Hz = int(h1_resonance_freq)
    hdr.experimentalConditions = exp

    # user parameters
    dtime = ismrmrd.xsd.userParameterDoubleType()
    dtime.name = 'dwellTime_us'
    dtime.value_ = dwell_time * 1e6
    hdr.userParameters = ismrmrd.xsd.userParametersType()
    hdr.userParameters.userParameterDouble.append(dtime)

    # encoding
    encoding = ismrmrd.xsd.encodingType()
    encoding.trajectory = ismrmrd.xsd.trajectoryType(traj_type)

    # set fov and matrix size
    efov = ismrmrd.xsd.fieldOfViewMm(m_to_mm(encoding_fov.x), m_to_mm(encoding_fov.y), m_to_mm(encoding_fov.z))
    rfov = ismrmrd.xsd.fieldOfViewMm(m_to_mm(recon_fov.x), m_to_mm(recon_fov.y), m_to_mm(recon_fov.z))

    ematrix = ismrmrd.xsd.matrixSizeType(int(encoding_matrix.n_x), int(encoding_matrix.n_y), int(encoding_matrix.n_z))
    rmatrix = ismrmrd.xsd.matrixSizeType(int(recon_matrix.n_x), int(recon_matrix.n_y), int(recon_matrix.n_z))

    # set encoded and recon spaces
    escape = ismrmrd.xsd.encodingSpaceType()
    escape.matrixSize = ematrix
    escape.fieldOfView_mm = efov
    rspace = ismrmrd.xsd.encodingSpaceType()
    rspace.matrixSize = rmatrix
    rspace.fieldOfView_mm = rfov
    encoding.encodedSpace = escape
    encoding.reconSpace = rspace

    # encoding limits
    limits = ismrmrd.xsd.encodingLimitsType()
    limits.kspace_encoding_step_1 = ismrmrd.xsd.limitType(k1_limits.min, k1_limits.max, k1_limits.center)
    limits.kspace_encoding_step_2 = ismrmrd.xsd.limitType(k2_limits.min, k2_limits.max, k2_limits.center)
    limits.slice = ismrmrd.xsd.limitType(slice_limits.min, slice_limits.max, slice_limits.center)
    limits.contrast = ismrmrd.xsd.limitType(contrast_limits.min, contrast_limits.max, contrast_limits.center)
    limits.average = ismrmrd.xsd.limitType(average_limits.min, average_limits.max, average_limits.center)
    limits.repetition = ismrmrd.xsd.limitType(repetition_limits.min, repetition_limits.max, repetition_limits.center)
    limits.phase = ismrmrd.xsd.limitType(phase_limits.min, phase_limits.max, phase_limits.center)
    limits.set = ismrmrd.xsd.limitType(set_limits.min, set_limits.max, set_limits.center)
    encoding.encodingLimits = limits

    # append encoding
    hdr.encoding.append(encoding)

    return hdr


def read_ismrmrd_dataset(fname: Path) -> tuple:
    """Read ismrmrd dataset from file.

    Parameters
    ----------
    fname
        file path to the ismrmrd dataset.

    Returns
    -------
        ismrmrd header and list of acquisitions.
    """
    with ismrmrd.File(str(fname), 'r') as file:
        ds = file[list(file.keys())[-1]]
        header = ds.header
        acqs = ds.acquisitions[:]

    return header, acqs


def insert_traj_from_meta(
    data_acqs: list[ismrmrd.acquisition.Acquisition],
    meta_acqs: list[ismrmrd.acquisition.Acquisition],
) -> list[ismrmrd.acquisition.Acquisition]:
    """
    Insert trajectory information from the meta file into the data file.

    Parameters
    ----------
    data_acqs : list
        list of acquisitions from the data file.
    meta_acqs : list
        list of acquisitions from the meta file.

    Returns
    -------
    list of acquisitions with the trajectory information from the meta file.
    """
    if not (data_len := len(data_acqs)) == (meta_len := len(meta_acqs)):
        raise ValueError(f'Number of acquisitions in data ({data_len}) and meta ({meta_len}) file do not match.')

    for i, (acq_d, acq_m) in enumerate(zip(data_acqs, meta_acqs, strict=False)):
        if not acq_d.number_of_samples == acq_m.number_of_samples:
            raise ValueError(f'Number of samples in acquisition {i} do not match.')

        # insert trajectory information from meta file
        acq_d.resize(
            number_of_samples=acq_d.number_of_samples,
            active_channels=acq_d.active_channels,
            trajectory_dimensions=acq_m.trajectory_dimensions,
        )
        acq_d.traj[:] = acq_m.traj[:]
        data_acqs[i] = acq_d

    return data_acqs


def update_header_from_meta(
    data_header: ismrmrd.xsd.ismrmrdHeader,
    meta_header: ismrmrd.xsd.ismrmrdHeader,
    enc_idx: int = 0,
) -> ismrmrd.xsd.ismrmrdHeader:
    """Update the header of the data file with the information from the meta file.

    Parameters
    ----------
    data_header : ismrmrd.xsd.ismrmrdHeader
        Header of the ISMRMRD data file.
    meta_header : ismrmrd.xsd.ismrmrdHeader
        Header of the ISMRMRD meta file created with the seq-file.
    enc_idx : int, optional
        Encoding index, by default 0

    Returns
    -------
    ismrmrd.xsd.ismrmrdHeader
        Updated header.
    """

    # Helper function to copy attributes if they are not None
    def copy_attributes(source, target, attributes):
        for attr in attributes:
            value = getattr(source, attr, None)
            if value is not None:
                setattr(target, attr, value)

    # Define the attributes to update for encodedSpace and reconSpace
    attributes_to_update = ['matrixSize', 'fieldOfView_mm']

    # Update encodedSpace
    copy_attributes(
        meta_header.encoding[enc_idx].encodedSpace,
        data_header.encoding[enc_idx].encodedSpace,
        attributes_to_update,
    )

    # Update reconSpace
    copy_attributes(
        meta_header.encoding[enc_idx].reconSpace,
        data_header.encoding[enc_idx].reconSpace,
        attributes_to_update,
    )

    # Update trajectory type
    if meta_header.encoding[enc_idx].trajectory is not None:
        data_header.encoding[enc_idx].trajectory = meta_header.encoding[enc_idx].trajectory

    return data_header


def combine_ismrmrd_files(data_file: Path, meta_file: Path, filename_ext: str = '_with_traj.mrd') -> ismrmrd.Dataset:
    """Combine ismrmrd data file and meta file.

    Parameters
    ----------
    data_file
        path to the ismrmrd data file
    meta_file
        path to the ismrmrd meta file
    filename_ext
        filename extension of the output file

    Returns
    -------
        combined ismrmrd file from data and meta file.
    """
    filename_out = data_file.parent / (data_file.stem + filename_ext)

    data_header, data_acqs = read_ismrmrd_dataset(data_file)
    meta_header, meta_acqs = read_ismrmrd_dataset(meta_file)

    new_acqs = insert_traj_from_meta(data_acqs, meta_acqs)
    new_header = update_header_from_meta(data_header, meta_header)

    # Create new file
    ds = ismrmrd.Dataset(filename_out)
    ds.write_xml_header(new_header.toXML())

    # add acquisitions with trajectory information
    for acq in new_acqs:
        ds.append_acquisition(acq)

    ds.close()

    return ds


def ismrmrd_from_sequence(adc_data_list: Sequence[np.ndarray], filename_seq: str, filename_mrd: str) -> ismrmrd.Dataset:
    """Create ismrmrd file based on list of adc data and pulseq sequence file.

    Parameters
    ----------
    adc_data_list
        list of numpy arrays where each array is one acquisition raw data block
    filename_seq
        filename for sequence file
    filename_mrd
        filename for output ISMRMRD file

    Returns
    -------
       ISMRMRD raw data file
    """
    sequence = pp.Sequence()
    sequence.read(filename_seq)

    adc_labels = sequence.evaluate_labels(evolution='adc')
    # Make labels into lists rather than numpy arrays because ismrmrd cannot deal well with numpy
    for key in adc_labels:
        adc_labels[key] = adc_labels[key].tolist()

    if (n_labels := len(adc_labels.get('LIN', 0))) != (n_adc_data := len(adc_data_list)):
        raise ValueError(f'Number of acquisitions ({n_adc_data}) and labels ({n_labels}) do not match.')

    # Get adc dwell time
    adc_blocks = [sequence.get_block(be).adc for be in sequence.block_events if sequence.get_block(be).adc is not None]

    readout_oversampling = (
        sequence.get_definition('ReadoutOversamplingFactor')
        if sequence.get_definition('ReadoutOversamplingFactor')
        else 1.0
    )

    # Create new file
    ds = ismrmrd.Dataset(filename_mrd, create_if_needed=True)

    n_readout = adc_data_list[0].shape[-1]
    num_channels = adc_data_list[0].shape[-2]
    n_phase_encoding = max(adc_labels.get('LIN', 0)) - min(adc_labels.get('LIN', 0)) + 1
    n_slice_encoding = max(adc_labels.get('PAR', 0)) - min(adc_labels.get('PAR', 0)) + 1
    hdr = create_header(
        traj_type='cartesian',
        encoding_fov=Fov(*sequence.get_definition('FOV').tolist()),
        recon_fov=Fov(*sequence.get_definition('FOV').tolist()),
        encoding_matrix=MatrixSize(n_x=n_readout, n_y=n_phase_encoding, n_z=n_slice_encoding),
        recon_matrix=MatrixSize(n_x=int(n_readout / readout_oversampling), n_y=n_phase_encoding, n_z=n_slice_encoding),
        dwell_time=adc_blocks[0].dwell,
        k1_limits=Limits.from_label_list(adc_labels.get('LIN', (0,))),
        k2_limits=Limits.from_label_list(adc_labels.get('PAR', (0,))),
        slice_limits=Limits.from_label_list(adc_labels.get('SLC', (0,))),
        contrast_limits=Limits.from_label_list(adc_labels.get('ECO', (0,))),
        average_limits=Limits.from_label_list(adc_labels.get('AVG', (0,))),
        repetition_limits=Limits.from_label_list(adc_labels.get('REP', (0,))),
        phase_limits=Limits.from_label_list(adc_labels.get('PHS', (0,))),
        set_limits=Limits.from_label_list(adc_labels.get('SET', (0,))),
        h1_resonance_freq=sequence.system.gamma * sequence.system.B0,
    )

    def get_sequence_definition(sequence, definition_parameter: str, value_scaling: float = 1.0):
        value = sequence.get_definition(definition_parameter)
        if isinstance(value, np.ndarray):
            return [val.item() for val in (value_scaling * value)]
        if isinstance(value, np.generic):
            return [value_scaling * float(value)]
        if len(value) == 0:
            return []
        return value

    # Sequence Information
    seq = ismrmrd.xsd.sequenceParametersType()
    seq.TR = get_sequence_definition(sequence, 'TR', 1e3)
    seq.TE = get_sequence_definition(sequence, 'TE', 1e3)
    seq.TI = get_sequence_definition(sequence, 'TI', 1e3)
    hdr.sequenceParameters = seq

    ds.write_xml_header(hdr.toXML())

    # add acquisitions with trajectory information
    for idx, adc_data in enumerate(adc_data_list):
        acq = ismrmrd.Acquisition()
        acq.resize(n_readout, num_channels)
        acq.data[:] = adc_data

        acq.center_sample = round(n_readout / 2)

        acq.idx.kspace_encode_step_1 = adc_labels.get('LIN')[idx] if 'LIN' in adc_labels else 0
        acq.idx.kspace_encode_step_2 = adc_labels.get('PAR')[idx] if 'PAR' in adc_labels else 0
        acq.idx.slice = adc_labels.get('SLC')[idx] if 'SLC' in adc_labels else 0
        acq.idx.contrast = adc_labels.get('ECO')[idx] if 'ECO' in adc_labels else 0
        acq.idx.repetition = adc_labels.get('REP')[idx] if 'REP' in adc_labels else 0
        acq.idx.phase = adc_labels.get('PHS')[idx] if 'PHS' in adc_labels else 0
        acq.idx.set = adc_labels.get('SET')[idx] if 'SET' in adc_labels else 0

        acq.read_dir = (1.0, 0.0, 0.0)
        acq.phase_dir = (0.0, 1.0, 0.0)
        acq.slice_dir = (0.0, 0.0, 1.0)

        # Flags
        if adc_labels.get('NAV', 0)[idx]:
            acq.setFlag(ismrmrd.ACQ_IS_PHASECORR_DATA)

        if adc_labels.get('NOISE', 0)[idx]:
            acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)

        ds.append_acquisition(acq)

    ds.close()

    return ds
