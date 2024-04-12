import os
import shutil
import globals

DIR_BIOGEARS_BIN = globals.DIR_BIOGEARS_BIN
source = os.path.join(DIR_BIOGEARS_BIN, "csv-data")
target = "asthma-dataset-1"

def copy_xml_files(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for root, dirs, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)
        os.makedirs(target_path, exist_ok=True)
        for file in files:
            if file.endswith('.csv'):
                shutil.copy2(os.path.join(root, file), target_path)

copy_xml_files(source, target)
