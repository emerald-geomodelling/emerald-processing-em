import unittest
import tempfile
import os, yaml, shutil
import filecmp
import libaarhusxyz
import pandas as pd
import emeraldprocessing.pipeline
import difflib

datdir = os.path.join(os.getcwd().split('emeraldprocessing')[0], 'demo_data')
project_crs=31984   # demo data is part of Ancieta


# in this test code a list of processing tasks is tested as
# 1) a processing flow: test class Pipeline
# 2) each processing task individually: tesk class: ProcessingSubTasks
#
# the input us a demo (subset) dataset from the Project survey in the
#  ../../demo_data/ subfolder
# this folder also constains required auciliary data for testing (DEM, shape file)
#
# the processing code is tested agains a desired output
# the desireed output is stored in
# ../../demo_data/output
# for both the entire pipeline (full_pipeline.xyz/.alc/.gex)
# and the individual subtasks (sub_task_name.xyz/.alc/.gex)
#
# the desired output was generated with a notebook called make_desired_output.ipynb
# .. in the same subfolder as this code


def read_file_without_comment(file, comment_str='/'):
    with open(file) as f:
        file_lines=f.readlines()
    file_lines_without_comments=[]
    
    for line in file_lines:
        if not line[0]=='/':
            file_lines_without_comments.append(line)
    return file_lines_without_comments

def compare_files_without_comments(file1, file2, comment_str='/'):
    file1_lines=read_file_without_comment(file1, comment_str=comment_str)
    file2_lines=read_file_without_comment(file2, comment_str=comment_str)
    if file1_lines == file2_lines:
        return True
    else:
        return False

def assert_df_equal(df1, df2, obj):
    extra = set(df1.columns) - set(df2.columns)
    missing = set(df2.columns) - set(df1.columns)
    assert (len(missing) == 0) and (len(extra) == 0), "Missing columns in " + obj + ": " + str(missing) + ", extra columns in " + obj + ": " + str(extra)
    pd.testing.assert_frame_equal(df1, df2, obj=obj)
    
def assert_xyz_equal(file1, file2):
    # Compare file1 to file2 (things in file2 but not in file1 is reported as missing)
    f1 = libaarhusxyz.XYZ(file1)
    f2 = libaarhusxyz.XYZ(file2)

    assert_df_equal(f1.flightlines, f2.flightlines, obj="flightlines")

    l1 = set(f1.layer_data.keys())
    l2 = set(f2.layer_data.keys())
    assert (len(l1 - l2) == 0) and (len(l2 - l1) == 0), "Extra layer data: " + str(l1 - l2) + ", missing layer data: " + str(l2 - l1)

    for key in f1.layer_data.keys():
        assert_df_equal(f1.layer_data[key], f2.layer_data[key], obj=key)
    
def get_input_file_names(datdir):
    infile_xyz=os.path.join(datdir,'Demo.xyz')
    infile_alc=os.path.join(datdir,'SkyTEM_LMZ_HMZ_STD_demo.alc')
    infile_gex=os.path.join(datdir,'Demo.gex')
    return infile_xyz, infile_alc, infile_gex

def get_test_file_names(tmpdir, prefix):
    test_xyz=os.path.join(tmpdir, prefix+'.xyz')
    test_alc=os.path.join(tmpdir, prefix+'.alc')
    test_gex=os.path.join(tmpdir, prefix+'.gex')
    return test_xyz, test_alc, test_gex

def get_desired_output_file_names(datdir, prefix):
    output_dir=os.path.join(datdir, 'output')
    output_xyz=os.path.join(output_dir, prefix+'.xyz')
    output_alc=os.path.join(output_dir, prefix+'.alc')
    output_gex=os.path.join(output_dir, prefix+'.gex')
    return output_xyz, output_alc, output_gex

def get_items_from_processing_list(processing_list, pattern):
    item_list=[]
    for item in processing_list:
        if pattern.__eq__(list(item.keys())[0].split('.')[-1]):
            item_list.append(item)
    if len(item_list)>0:
        return item_list
    else:
        raise Exception('no item with pattern ´{}´ found in processsing list'.format(pattern))



def reorder_xyz(xyz):
    colorder = ["fid", "Line", "Flight", "Date", "Time", "TxPitch", "TxRoll", "TxAltitude", "lon", "lat", "UTMX", "UTMY", "Topography", "Alt", "GdSpeed", "rmf", "Magnetic", "PowerLineMonitor", "Misc2", "Misc1", "Current_Ch02", "Current_Ch01", "DateTime", "xdist" ,"DipoleMoment_Ch01", "DipoleMoment_Ch02", "TxZ"]
    xyz.flightlines = xyz.flightlines[colorder + [col for col in xyz.flightlines.columns if col not in colorder]]

    layer_data = xyz.layer_data
    xyz.layer_data = {}
    layerorder = ["Gate_Ch01", "Gate_Ch02", "STD_Ch01", "STD_Ch02", "InUse_Ch01", "InUse_Ch02"]
    for col in layerorder:
        if col in layer_data:
            xyz.layer_data[col] = layer_data[col]
    for col in layer_data.keys():
        if col not in layerorder:
            xyz.layer_data[col] = layer_data[col]

def clean_xyz(xyz):
    del(xyz.model_info['scalefactor'])
    # xyz.model_info = {}
    # for key in ['xdist', 'DateTime']:
    #     if key in xyz.flightlines.columns: del xyz.flightlines[key]
    
    # for name in ["Flight"]:
    #     if name in xyz.flightlines.columns:
    #         xyz.flightlines.rename(columns={name: name.lower()}, inplace=True)


def test_processing_flow(tempdir, processing_list, output_prefix):
    os.chdir(os.getcwd().split('emeraldprocessing')[0])
    
    infile_xyz, infile_alc, infile_gex = get_input_file_names(datdir)
    
    d = emeraldprocessing.pipeline.ProcessingData(infile_xyz, infile_alc, infile_gex, crs=project_crs)

    d.process(processing_list)

    # order of keys in d.xyz.layer data can be arbitrary, sort before
    # writing to disk to get consistent xyz files
    reorder_xyz(d.xyz)
    
    # # FIXME: remove once we have updated the desired data files...
    # this is needed to use the same desired output in the dev and master branch
    #clean_xyz(d.xyz)
    
    test_xyz, test_alc, test_gex = get_test_file_names(tempdir, 'test')
    
    d.save(test_xyz, test_alc, test_gex)
    
    desired_xyz, desired_alc, desired_gex = get_desired_output_file_names(datdir, output_prefix)

    assert_xyz_equal(test_xyz, desired_xyz)
    if not filecmp.cmp(test_xyz, desired_xyz, shallow=False):
        with open(test_xyz) as f:
            test = f.readlines()
        with open(desired_xyz) as f:
            desired = f.readlines()
        raise Exception("XYZ differs:\n" + "\n".join(difflib.unified_diff(desired, test, fromfile='test.xyz', tofile='desired.xyz')))
    assert filecmp.cmp(test_xyz, desired_xyz, shallow=False), "xyz files differ from desired for: {}".format(output_prefix) # lets keep this code for a while
    assert filecmp.cmp(test_alc, desired_alc, shallow=False), "alc files differ from desired for: {}".format(output_prefix)
    assert compare_files_without_comments(test_gex, desired_gex, comment_str='/'), "gex files differ from desired for: {}".format(output_prefix)

def test_processing_task(tempdir, processing_yaml_file, task_name):
    with open(processing_yaml_file, 'r') as file:
        processing_list = yaml.safe_load(file)
    
    processing_sublist = get_items_from_processing_list(processing_list, task_name)
    print(f'\n\n=============== Test task: {task_name} ===============')
    print(f'  - Options:')
    print(f'      {processing_sublist}')
    
    test_processing_flow(tempdir, processing_sublist, task_name)

class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory().name
        os.mkdir(self.tempdir)
        print('\n using tmp dir: {}'.format(self.tempdir))
    
    def tearDown(self):
        shutil.rmtree(self.tempdir)
    
    def test_full_processing_pipeline(self):
        print('\n\n=============== Integration testing full workflow ===============')
        with open(os.path.join(datdir, 'processing_flow.yml'), 'r') as file:
            processing_list = yaml.safe_load(file)
            
        test_processing_flow(self.tempdir, processing_list, 'full_pipeline')

class ProcessingSubTasks(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory().name
        os.mkdir(self.tempdir)
        print('\n using tmp dir: {}'.format(self.tempdir))
    
    def tearDown(self):
        shutil.rmtree(self.tempdir)
        #print('remove temp folder manually: {}'.format(self.tempdir))
    
    # def test_individual_processing_tasks(self):
    #     with open(os.path.join(datdir, 'processing_flow.yml'), 'r') as file:
    #         processing_list = yaml.safe_load(file)
        
    #     for item in processing_list:
    #         task=list(item.keys())[0].split('.')[-1]
    #         processing_sublist = get_items_from_processing_list(processing_list, task)
    #         print('\n\n\n=================test task: {0}===============\n options: \n {1}\n'.format(task, processing_sublist))
            
    #         test_processing_flow(self.tempdir, processing_sublist, task)
            
    def test_correct_altitude_and_topo(self):
        test_processing_task(self.tempdir, 
                             os.path.join(datdir, 'processing_flow.yml'),
                             'correct_altitude_and_topo')
    
    def test_cull_roll_pitch_alt(self):
        test_processing_task(self.tempdir, 
                             os.path.join(datdir, 'processing_flow.yml'),
                             'cull_roll_pitch_alt')
    
    
    def test_cull_max_slope(self):
        test_processing_task(self.tempdir, 
                             os.path.join(datdir, 'processing_flow.yml'),
                             'cull_max_slope')
    
    def test_moving_average_filter(self):
        test_processing_task(self.tempdir, 
                             os.path.join(datdir, 'processing_flow.yml'),
                             'moving_average_filter')
    
    def test_cull_std_threshold(self):
        test_processing_task(self.tempdir, 
                             os.path.join(datdir, 'processing_flow.yml'),
                             'cull_std_threshold')
    
    def test_cull_on_geometry(self):
        test_processing_task(self.tempdir, 
                             os.path.join(datdir, 'processing_flow.yml'),
                             'cull_on_geometry')
    
    def test_cull_soundings_with_too_few_gates(self):
        test_processing_task(self.tempdir, 
                             os.path.join(datdir, 'processing_flow.yml'),
                             'cull_soundings_with_too_few_gates')
    
    def test_cull_below_noise_level(self):
        test_processing_task(self.tempdir, 
                             os.path.join(datdir, 'processing_flow.yml'),
                             'cull_below_noise_level')
    def test_cull_negative_data(self):
        test_processing_task(self.tempdir, 
                             os.path.join(datdir, 'processing_flow.yml'),
                             'cull_negative_data')
    
    def test_correct_tilt_pitch_for1D(self):
        test_processing_task(self.tempdir, 
                             os.path.join(datdir, 'processing_flow.yml'),
                             'correct_tilt_pitch_for1D')
    
    def test_cull_on_geometry_and_inversion_misfit(self):
        test_processing_task(self.tempdir, 
                             os.path.join(datdir, 'processing_flow.yml'),
                             'cull_on_geometry_and_inversion_misfit')
    
    def test_subtract_system_bias(self):
        test_processing_task(self.tempdir, 
                             os.path.join(datdir, 'processing_flow.yml'),
                             'subtract_system_bias')

