import os
import tarfile
import shutil

def extract_tar_files(source_dir, destination_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Iterate over files in the source directory
    for file_name in os.listdir(source_dir):
        if file_name.endswith('.tar.gz'):
            file_path = os.path.join(source_dir, file_name)
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(destination_dir)
            print(f"Extracted {file_name}")

if __name__ == "__main__":
    source_directory = "C:\\Users\\Abrar\\Downloads\\kvasir-capsule-labeled-images.zip\\labelled_images"
    destination_directory = "E:\BUET Files\Celia MAM Biomedical Signal Processing\RAtCapsNet\data\collected_data"

    extract_tar_files(source_directory, destination_directory)
