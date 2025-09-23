"""Tests for fat saturation pulse."""

import pypulseq as pp
import pytest
from mrseq.preparations.fat_sat import add_fat_sat


def test_add_fat_sat_system_defaults_if_none(system_defaults):
    """Test if system defaults are used if no system limits are provided."""
    _, block_duration1 = add_fat_sat(system=system_defaults)
    _, block_duration2 = add_fat_sat(system=None)

    assert block_duration1 == block_duration2


@pytest.mark.parametrize(
    ('rf_duration', 'rf_flip_angle', 'saturation_frequency_ppm'),
    [(8.0e-3, 110, -3.45), (8.0e-3, 160, -3.45), (8.0e-3, 110, -5.45), (10.0e-3, 110, -3.45)],
    ids=['defaults', 'higher flip angle', 'larger frequency offset', 'longer pulse'],
)
def test_add_t1_inv_prep_duration(system_defaults, rf_duration, rf_flip_angle, saturation_frequency_ppm):
    """Ensure the default parameters are set correctly."""
    seq = pp.Sequence(system=system_defaults)

    seq, block_duration = add_fat_sat(
        seq=seq,
        system=system_defaults,
        rf_duration=rf_duration,
        rf_flip_angle=rf_flip_angle,
        saturation_frequency_ppm=saturation_frequency_ppm,
    )

    assert sum(seq.block_durations.values()) == block_duration
