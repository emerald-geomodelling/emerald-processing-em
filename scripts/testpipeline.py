import emeraldprocessing.pipeline
import os

data_dir = <Path to file directory>
filename_xyz_in = <filename.xyz>
fullfilename_xyz_in = os.path.join(data_dir, filename_xyz_in)

filename_alc_in = <filename.alc>
fullfilename_alc = os.path.join(data_dir, filename_alc_in)

filename_gex_in = <filename.gex>
fullfilename_gex = os.path.join(data_dir, filename_gex_in)


d = emeraldprocessing.pipeline.ProcessingData(fullfilename_xyz_in,
                                              fullfilename_alc,
                                              fullfilename_gex)

d.process([
    {"name": "emeraldprocessing.pipeline.cull_roll_pitch_alt", "args": {
        "max_roll": 12,
        "max_pitch": 12,
        "max_alt": 110}},
    }
])

d.save("test.xyz", "test.alc", "test.gex")
