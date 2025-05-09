import os
import shutil
from sys import argv

def check_and_copy(path):
    if os.path.exists(path):
        print("Path already exists.")
    else:
        # Split the input path to get the parent directory
        parent_dir = os.path.dirname(path)
        print(parent_dir)
        if os.path.exists(parent_dir):
            # Get the current working directory
            cwd = os.getcwd()
            # Construct the source path
            source_path = os.path.join(cwd, "../ws_template")
            # Copy the source path to the destination path
            shutil.copytree(source_path, path)
        else:
            print("Parent directory does not exist.")

# Test the function
path = argv[1]
check_and_copy(path)