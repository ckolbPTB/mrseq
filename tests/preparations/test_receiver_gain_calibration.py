"""Tests for the receiver gain calibration block."""

from mrseq.preparations.receiver_gain_calibration import add_gre_receiver_gain_calibration


def test_add_gre_rx_gain_calib_system_defaults_if_none(system_defaults):
    """Test if system defaults are used if no system limits are provided."""
    _, block_duration1 = add_gre_receiver_gain_calibration(system=system_defaults)
    _, block_duration2 = add_gre_receiver_gain_calibration(system=None)

    assert block_duration1 == block_duration2
