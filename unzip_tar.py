import os
import tarfile

# Directory containing .tar.gz files
directory = r'E:\\BUET Files\\Celia MAM Biomedical Signal Processing\\RAtCapsNet\\data\\labelled_images'

# Function to extract .tar.gz files
def extract_tar_gz(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Iterate through each file
    for file in files:
        # Check if the file is a .tar.gz file
        if file.endswith('.tar.gz'):
            file_path = os.path.join(directory, file)
            # Open the .tar.gz file
            with tarfile.open(file_path, 'r:gz') as tar:
                # Extract the contents to the same directory
                tar.extractall(path=directory)
                print("Extracted {} to {}".format(file, directory))

# Call the function
extract_tar_gz(directory)