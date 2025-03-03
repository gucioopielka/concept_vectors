import yaml
import os

# Define the path to the globals file
base_dir = os.path.dirname(os.path.abspath(__file__))
globals_path = os.path.join(base_dir, '../../', 'globals.yml')

# Load data from YAML
with open(globals_path, "r") as stream:
    data = yaml.safe_load(stream)

# Define the absolute paths to the data files
FILES = [os.path.join(base_dir, '../../', data[file]) for file in data.keys() if 'FILE' in file or 'DIR' in file]
RESULTS_DIR, DATASET_DIR, PLOTS_DIR = FILES