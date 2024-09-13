#!/usr/bin/env python

import setuptools


setuptools.setup(
    name='emeraldprocessing',
    version='0.1.3',
    description="""Processing tools for multi-sounding EM geophysical data""",
    long_description="""Processing tools and and workflow for multi-sounding EM geophysical data.""",
    long_description_content_type="text/markdown",
    authors='Benjamin R. Bloss, Egil Möller, Martin Panzner',
    author_emails='bb@emrld.no, em@emrld.no, mp@emrld.no',
    url='https://github.com/emerald-geomodelling/emerald-processing-em',
    packages=setuptools.find_packages(),
    install_requires=[
        "libaarhusxyz",
        "matplotlib",
        "geopandas",
        "scipy",
        "mercantile",
        "requests",
        "mapbox_vector_tile",
        "nose2",
        "pyyaml",
        "rasterio",
        "contextily",
        "pydantic",
        "projnames",
        "python-slugify"
    ],
    entry_points = {
        'emeraldprocessing.pipeline_step': [
            'Save intermediate results=emeraldprocessing.pipeline:save_intermediate',
            'Use manual edits=emeraldprocessing.diff:apply_diff',
            # Corrections:
            'Classify high altitude flightlines: Auto=emeraldprocessing.tem.corrections:auto_classify_high_altitude_flightlines',
            'Classify flightlines: Selection=emeraldprocessing.tem.corrections:classify_flightlines',
            'Select flightlines=emeraldprocessing.tem.corrections:select_lines',
            'Select flight-types=emeraldprocessing.tem.corrections:select_flight_types',
            'Correct altitude and topo=emeraldprocessing.tem.corrections:correct_altitude_and_topo',
            'Correct data and tilt for 1D=emeraldprocessing.tem.corrections:correct_data_tilt_for1D',
            'Moving average filter=emeraldprocessing.tem.corrections:moving_average_filter',
            'STD error: Add fractional error=emeraldprocessing.tem.corrections:add_std_error',
            'STD error: Replace from GEX=emeraldprocessing.tem.corrections:add_replace_gex_std_error',
            'Rename Column=emeraldprocessing.tem.corrections:rename_column',
            'Rename Data=emeraldprocessing.tem.corrections:rename_data',
            # Cullings:
            'Apply gex=emeraldprocessing.tem.culling:apply_gex',
            'Enable/Disable gates by index=emeraldprocessing.tem.culling:enable_disable_time_gate',
            'Disable soundings by tilt and altitude=emeraldprocessing.tem.culling:cull_roll_pitch_alt',
            'Disable soundings by number of active gates=emeraldprocessing.tem.culling:cull_soundings_with_too_few_gates',
            'Disable gates by STD values=emeraldprocessing.tem.culling:cull_std_threshold',
            'Disable gates by slope max=emeraldprocessing.tem.culling:cull_max_slope',
            'Disable gates by slope min=emeraldprocessing.tem.culling:cull_min_slope',
            'Disable gates by curvature max=emeraldprocessing.tem.culling:cull_max_curvature',
            'Disable gates by curvature min=emeraldprocessing.tem.culling:cull_min_curvature',
            'Disable gates by noise floor=emeraldprocessing.tem.culling:cull_below_noise_level',
            'Disable gates by geometry=emeraldprocessing.tem.culling:cull_on_geometry',
            'Disable gates by geometry and misfit=emeraldprocessing.tem.culling:cull_on_geometry_and_inversion_misfit',
            'Disable gates by negative data=emeraldprocessing.tem.culling:cull_negative_data',
            # System Bias:
            # FIXME:  as is 'Subtract System Bias' cannot run without having built the system_response yaml first./
            #   Need to incorporate the yaml building here first./
            #   https://github.com/emerald-geomodelling/EmeraldProcessing/tree/40515c839524eae748db3db5d279b747df87de30/notebooks/TEM/EstimateSystemBias
            # 'Subtract System Bias=emeraldprocessing.tem.system_bias:subtract_system_bias',
        ],
    }
)
