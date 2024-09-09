from . import pipeline
import typing
import libaarhusxyz
import libaarhusxyz.export.msgpack
import pydantic

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
        
    diffxyz.normalize_naming(naming_standard="alc")

    processing.xyz = processing.xyz.apply_diff(diffxyz)

