import logging
import os

home_dir = os.environ['HOME']

current_script_location = os.path.realpath(__file__)
current_dir = os.path.dirname(current_script_location)
base_project_dir = os.path.dirname(current_dir)

logging.getLogger('matplotlib').setLevel(logging.INFO)