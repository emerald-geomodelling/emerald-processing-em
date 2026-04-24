"""Tests for (fid, Line) composite-key apply_idx remapping.

Sprint 08 v2: Verifies that _remap_apply_idx_by_fid_line correctly
resolves (fid, Line) pairs to current positional indices, making diffs
immune to global sort-order changes. Also verifies column-drop safety.
"""

import sys
import types
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

_mock_pipeline = types.ModuleType("emeraldprocessing.pipeline")
_mock_pipeline.ProcessingData = MagicMock
sys.modules.setdefault("emeraldprocessing.pipeline", _mock_pipeline)

from emeraldprocessing.diff import _remap_apply_idx_by_fid_line


def _make_xyz(fids, lines):
    fl = pd.DataFrame({"fid": fids, "Line": lines})
    xyz = MagicMock()
    xyz.flightlines = fl
    xyz.line_id_column = "Line"
    return xyz


def _make_diff(apply_idx, fids=None, lines=None, inuse_vals=None):
    fl = pd.DataFrame({"apply_idx": np.array(apply_idx, dtype=np.int64)})
    if fids is not None:
        fl["fid"] = fids
    if lines is not None:
        fl["Line"] = lines
    if inuse_vals is not None:
        fl["InUse_Ch01"] = inuse_vals
    diff = MagicMock()
    diff.flightlines = fl
    return diff


class TestRemapCompositeKey:

    def test_no_fid_is_noop(self):
        target = _make_xyz([100.0, 200.0], [1, 1])
        diff = _make_diff([0, 1])
        _remap_apply_idx_by_fid_line(target, diff)
        np.testing.assert_array_equal(diff.flightlines["apply_idx"].values, [0, 1])

    def test_fid_only_without_line_is_noop(self):
        """v1 diffs with fid but no Line should NOT trigger remapping."""
        target = _make_xyz([100.0, 200.0], [1, 1])
        diff = _make_diff([0, 1], fids=[100.0, 200.0])
        _remap_apply_idx_by_fid_line(target, diff)
        np.testing.assert_array_equal(diff.flightlines["apply_idx"].values, [0, 1])

    def test_composite_remapping(self):
        """(fid, Line) correctly remaps when sort order differs."""
        target = _make_xyz([300.0, 100.0, 200.0], [2, 1, 1])
        diff = _make_diff([0, 2], fids=[100.0, 300.0], lines=[1, 2])
        _remap_apply_idx_by_fid_line(target, diff)
        np.testing.assert_array_equal(diff.flightlines["apply_idx"].values, [1, 0])

    def test_duplicate_fids_resolved_by_line(self):
        """When same fid appears on different lines, composite key picks the right one."""
        target = _make_xyz(
            [100.0, 100.0, 200.0],  # fid 100 on BOTH line 1 and line 2
            [1, 2, 1]
        )
        # Want fid=100 on line 2 (position 1), not line 1 (position 0)
        diff = _make_diff([99], fids=[100.0], lines=[2])
        _remap_apply_idx_by_fid_line(target, diff)
        np.testing.assert_array_equal(diff.flightlines["apply_idx"].values, [1])

    def test_identical_order_unchanged(self):
        target = _make_xyz([100.0, 200.0, 300.0], [1, 1, 2])
        diff = _make_diff([0, 2], fids=[100.0, 300.0], lines=[1, 2])
        _remap_apply_idx_by_fid_line(target, diff)
        np.testing.assert_array_equal(diff.flightlines["apply_idx"].values, [0, 2])

    def test_missing_key_falls_back(self):
        target = _make_xyz([100.0, 200.0], [1, 1])
        diff = _make_diff([0], fids=[999.0], lines=[1])
        _remap_apply_idx_by_fid_line(target, diff)
        # Fallback: apply_idx unchanged
        np.testing.assert_array_equal(diff.flightlines["apply_idx"].values, [0])

    def test_preserves_other_columns(self):
        target = _make_xyz([300.0, 100.0], [2, 1])
        diff = _make_diff([0], fids=[100.0], lines=[1], inuse_vals=[0])
        _remap_apply_idx_by_fid_line(target, diff)
        np.testing.assert_array_equal(diff.flightlines["InUse_Ch01"].values, [0])
        np.testing.assert_array_equal(diff.flightlines["apply_idx"].values, [1])

    def test_large_shift_with_duplicates(self):
        """Simulate real scenario: 200 soundings, duplicate fids, 54-position shift."""
        n = 200
        fids = np.tile(np.arange(100.0, 200.0), 2)  # 100 unique fids, each on 2 lines
        lines = np.array([1]*100 + [2]*100)

        # Shift: first 54 elements move to end
        new_order = np.concatenate([np.arange(54, n), np.arange(0, 54)])
        target = _make_xyz(fids[new_order], lines[new_order])

        # Diff targets positions 60-65 in old order (all on line 1)
        diff_positions = [60, 61, 62, 63, 64, 65]
        diff = _make_diff(
            diff_positions,
            fids=fids[diff_positions],
            lines=lines[diff_positions],
        )

        _remap_apply_idx_by_fid_line(target, diff)

        remapped = diff.flightlines["apply_idx"].values
        for i, orig_pos in enumerate(diff_positions):
            expected_fid = fids[orig_pos]
            expected_line = lines[orig_pos]
            actual_fid = fids[new_order[remapped[i]]]
            actual_line = lines[new_order[remapped[i]]]
            assert expected_fid == actual_fid and expected_line == actual_line

    def test_columns_not_dropped_by_remap(self):
        """The remap function should NOT drop fid/Line — that's the caller's job."""
        target = _make_xyz([100.0], [1])
        diff = _make_diff([0], fids=[100.0], lines=[1])
        _remap_apply_idx_by_fid_line(target, diff)
        # fid and Line should STILL be present (caller drops them)
        assert "fid" in diff.flightlines.columns
        assert "Line" in diff.flightlines.columns

    def test_columns_survive_fallback(self):
        """On fallback, fid/Line stay in the diff (caller still responsible for drop)."""
        target = _make_xyz([100.0], [1])
        diff = _make_diff([0], fids=[999.0], lines=[1])  # fid not found
        _remap_apply_idx_by_fid_line(target, diff)
        assert "fid" in diff.flightlines.columns
        assert "Line" in diff.flightlines.columns
