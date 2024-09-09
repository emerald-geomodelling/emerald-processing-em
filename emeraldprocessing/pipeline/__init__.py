import libaarhusxyz
import importlib
import pandas as pd
import numpy as np
import tempfile
import typing
import pydantic
import time
import os
import importlib
import yaml
import copy
import slugify
from ..tem import dataIO

processing_steps = {entry.name: entry for entry in importlib.metadata.entry_points()['emeraldprocessing.pipeline_step']}    

def load_fn(name):
    return processing_steps[name].load()
    
class ProcessingData(libaarhusxyz.Survey):
    json_schema = {"hide": True}
    
    def __init__(self, data, sidecar=None, system_data=None, crs=None, outdir = None):
        self.outdir = outdir
        if isinstance(data, libaarhusxyz.XYZ):
            xyz = data
            gex = system_data
        else:
            xyz, gex = dataIO.readSkyTEMxyz(data, sidecar, system_data)
        libaarhusxyz.Survey.__init__(self, xyz, gex)
        dataIO.makeGateTimesDipoleMoments(self)

        if crs is not None:
            self.crs = crs
        else:
            self.crs = self.xyz.model_info['projection']

    def process(self, steps):
        """steps is a list of processing steps
        steps = [{"modulename.function_name": {"arg1": "value1", ...}}, ...]

        Note: Each top level dictionary can only have a single
        function listed.

        modulename.function_name must be a function that takes a
        ProcessingData instance as first argument, and args as keyword
        arguments. It must modify the ProcessingData instance in place.

        """
        pd.options.mode.chained_assignment = None  # get rid of warning 'A value is trying to be set on a copy of a slice from a DataFrame'
        
        for idx, step in enumerate(steps):
            name, args = next(iter(step.items()))
            print(f"\n=============== Processing step {idx+1} ===============")
            print(f"  - Step name: '{name}'")
            load_fn(name)(self, **args)

        self.clean_xyz()

    def clean_xyz(self):
        # do some cleanup prior to dumping to file
        columns_to_strip=['cull_reason', 'coverage', 'geometry']
        for column in columns_to_strip:
            if column in self.xyz.flightlines.columns:
                # strip auxiliary columns from dataframe before writing to disk
                self.xyz.flightlines.drop(column, inplace=True, axis=1)
    
    def save(self, datafile, alcfile=None, gexfile=None, msgpackfile=None):
        assert gexfile is not None, "Please specify a gexfile"
        self.dump(xyzfile=datafile, gexfile=gexfile, alcfile=alcfile, msgpackfile=msgpackfile)
        
def save_intermediate(processing: ProcessingData,
                      name: str):

    """
    Save an intermediate state of your new processing.
    
    Parameters
    ----------
    name : 
        Give the intermediate result a dataset name. This will appear in the plot editor under Dataset.
 
    """

    assert processing.outdir is not None

    processing.dump(
        xyzfile = '%s/%s.xyz' % (processing.outdir, name),
        gexfile = '%s/%s.gex' % (processing.outdir, name),
        msgpackfile = '%s/%s.msgpack' % (processing.outdir, name),
        diffmsgpackfile = '%s/%s.diff.msgpack' % (processing.outdir, name),
        summaryfile = '%s/%s.summary.yml' % (processing.outdir, name),
        geojsonfile = '%s/%s.geojson' % (processing.outdir, name))

    for fline, line_data in processing.xyz.split_by_line().items():
        sfline = slugify.slugify(str(fline), separator="_")
        fl_processing = copy.copy(processing)
        fl_processing.xyz = line_data
        fl_processing.orig_xyz = processing.orig_xyz_by_line[fline]
        fl_processing.dump(
            xyzfile = '%s/%s.%s.xyz' % (processing.outdir, name, sfline),
            gexfile = '%s/%s.%s.gex' % (processing.outdir, name, sfline),
            msgpackfile = '%s/%s.%s.msgpack' % (processing.outdir, name, sfline),
            diffmsgpackfile = '%s/%s.%s.diff.msgpack' % (processing.outdir, name, sfline),
            summaryfile = '%s/%s.%s.summary.yml' % (processing.outdir, name, sfline),
            geojsonfile = '%s/%s.%s.geojson' % (processing.outdir, name, sfline))

if typing.TYPE_CHECKING:
    # from ..tem.corrections import correct_altitude_and_topo
    p = ProcessingData("data.xyz", "data.alc", "data.gex")
    correct_altitude_and_topo(p, "test.tiff")
