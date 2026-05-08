import os
import shutil

# Define the path to the folder containing the TIFF files
source_folder = r"F:\111025_vGat DMS DLS\Sample 6"
# Define the names for the subfolders
subfolders = {
    'ch0': os.path.join(source_folder, 'ch0'),
    'ch1': os.path.join(source_folder, 'ch1'),
    'ch2': os.path.join(source_folder, 'ch2'),
    'ch3': os.path.join(source_folder, 'ch3')
}

# Create the subfolders if they don't exist
for subfolder in subfolders.values():
    os.makedirs(subfolder, exist_ok=True)

# Iterate over the files in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        # Check the file name and move it to the appropriate subfolder
        if 'ch0' in filename:
            shutil.move(os.path.join(source_folder, filename), subfolders['ch0'])
        elif 'ch1' in filename:
            shutil.move(os.path.join(source_folder, filename), subfolders['ch1'])
        elif 'ch2' in filename:
            shutil.move(os.path.join(source_folder, filename), subfolders['ch2'])
        elif 'ch3' in filename:
            shutil.move(os.path.join(source_folder, filename), subfolders['ch3'])

print("Files have been separated into subfolders successfully.")
