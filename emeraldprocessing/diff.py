from . import pipeline
import logging
import typing
import libaarhusxyz
import libaarhusxyz.export.msgpack
import numpy as np
import pydantic

logger = logging.getLogger(__name__)

ManualEditUrl = typing.Annotated[
    typing.Any,
    {"json_schema": {
        "x-reference": "manual-edit",
        "anyOf": [
            {"x-url-media-type": "application/x-geophysics-xyz-model"},
            {"type": "object",
             "additionalProperties": False,
             "required": ["url"],
             "properties": {
                 "url": {"x-url-media-type": "application/x-geophysics-xyz-model"},
                 "title": {"type": "string"},
                 "id": {"type": "integer"}
             }}
        ]
    }}]

def apply_diff(processing : pipeline.ProcessingData, 
               diff: ManualEditUrl):

    """
    Apply a manual culling to your dataset.
    
    Parameters
    ----------
    diff : 
        Manual culling to apply. To create manual culling, save culling in plot workspace first and it will appear here.
    """

    if isinstance(diff, dict): diff = diff["url"]

    if diff.endswith(".xyz") or diff.endswith(".xyzd"):
        diffxyz = libaarhusxyz.XYZ(diff, normalize=False)
    elif diff.endswith(".msgpack"):
        diffxyz = libaarhusxyz.export.msgpack.load(diff)
        
    # Only normalize full XYZ files (.xyz/.xyzd), never diffs from msgpack.
    # Diffs have sparse model_dicts with raw values (not DataFrames) that
    # crash normalize_naming, and their column names already match the source.
    if diff.endswith(".xyz") or diff.endswith(".xyzd"):
        if hasattr(diffxyz, 'model_dict') and 'model_info' in diffxyz.model_dict:
            diffxyz.normalize_naming(naming_standard="alc")

    # Ensure diff_dummy is set — frontend manual edit diffs always use -1 as
    # the sentinel for "no change". If model_info is missing or incomplete
    # (e.g. from a load→re-save cycle), default it so apply_diff skips sentinels.
    # Note: model_info is a property, so set via model_dict directly.
    if not diffxyz.model_info.get("diff_dummy"):
        mi = diffxyz.model_dict.get("model_info", {})
        mi["diff_dummy"] = -1
        diffxyz.model_dict["model_info"] = mi

    _remap_apply_idx_by_fid_line(processing.xyz, diffxyz)

    # Always strip fid/Line from the diff before apply_diff so that
    # df_apply() doesn't overwrite these columns in the target dataset.
    # This must happen regardless of whether remapping succeeded or fell back.
    for col in ("fid", "Line"):
        if col in diffxyz.flightlines.columns:
            diffxyz.flightlines.drop(col, axis=1, inplace=True)

    processing.xyz = processing.xyz.apply_diff(diffxyz)


def _remap_apply_idx_by_fid_line(target_xyz, diffxyz):
    """Remap apply_idx using (fid, Line) composite key.

    When a dataset is reimported and libaarhusxyz.normalize() produces a
    different global sort order (e.g. NaN dates becoming valid dates),
    positional apply_idx values point to wrong soundings.  If the diff
    includes both ``fid`` and ``Line`` columns (added by the frontend
    since Sprint 08 v2), this function resolves each (fid, Line) pair to
    its current position in the target dataset and overwrites apply_idx.

    fid alone is not unique (18.6% duplicates across flight lines with
    the same time-of-day).  The composite key (fid, Line) is unique.

    No-op when the diff lacks both columns (old diffs, backwards compat).
    No-op when only fid is present without Line (v1 diffs — fid-only
    matching is broken due to duplicates, positional is safer).
    Falls back to positional apply_idx if any key is not found in target.
    """
    if "fid" not in diffxyz.flightlines.columns:
        return
    if "Line" not in diffxyz.flightlines.columns:
        return

    diff_fids = diffxyz.flightlines["fid"].values
    diff_lines = diffxyz.flightlines["Line"].values

    # Resolve the target's line column name (may be "Line", "title", etc.)
    target_line_col = getattr(target_xyz, "line_id_column", "Line")
    if target_line_col not in target_xyz.flightlines.columns:
        target_line_col = "Line"
    if target_line_col not in target_xyz.flightlines.columns:
        logger.warning("No line column in target dataset; skipping remap")
        return

    target_fids = target_xyz.flightlines["fid"].values
    target_lines = target_xyz.flightlines[target_line_col].values

    # Build (fid, Line) → positional-index lookup from the target dataset
    key_to_pos = {}
    for pos in range(len(target_fids)):
        key = (float(target_fids[pos]), int(target_lines[pos]))
        if key not in key_to_pos:
            key_to_pos[key] = pos

    # Remap each diff (fid, Line) to its current position in the target
    remapped = []
    for i in range(len(diff_fids)):
        key = (float(diff_fids[i]), int(diff_lines[i]))
        pos = key_to_pos.get(key)
        if pos is None:
            logger.warning(
                "(fid=%.1f, Line=%d) from diff not found in target; "
                "falling back to positional apply_idx",
                key[0], key[1],
            )
            return  # Abort remapping — use original apply_idx as-is
        remapped.append(pos)

    diffxyz.flightlines["apply_idx"] = np.array(remapped, dtype=np.int64)

