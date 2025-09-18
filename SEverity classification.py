#!/usr/bin/env python
# coding: utf-8

# In[80]:


import os
import shutil
from PIL import Image
import numpy as np
from skimage import morphology
from scipy.ndimage import binary_closing, binary_opening, label as nd_label
import imageio.v2 as imageio
import os
print("Current Working Directory:", os.getcwd())

# Directory containing the images

# ✅ Source directory (contains PNG files)
source_dir ='/home/priti17491/priti/Source'
# ✅ Destination directory (subfolders for severity)
destination_dir ='Destination'

# Create destination folders
output_dirs = {
    'low': os.path.join(destination_dir, 'low'),
    'medium': os.path.join(destination_dir, 'medium'),
    'high': os.path.join(destination_dir, 'high'),
    'no_crack': os.path.join(destination_dir, 'no_crack')
}

for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Only get PNG files from source_dir
for image_name in [f for f in os.listdir(source_dir) if f.endswith('.png')]:
    image_path = os.path.join(source_dir, image_name)
    print("Processing:", image_name)
    img = imageio.imread(image_path, pilmode='L')
    img_size = img.size
    img_bnr = (img > 0).astype(np.uint8)
    img_bnr_closed = binary_closing(img_bnr)
    img_bnr_opened = binary_opening(img_bnr_closed)


    img_labels, num_labels = nd_label(img_bnr_opened)


    if num_labels == 0:
            severity = "no_crack"

            shutil.move(image_path, os.path.join(output_dirs[severity], image_name))
            print(f'{image_name} contains no crack and has been labeled as {severity}.')
            continue  # Skip the rest of the loop for this image


    labels = range(1, num_labels + 1)
    sizes = np.array([np.sum(img_labels == lbl) for lbl in labels])


    order = np.argsort(sizes)[::-1]
    sorted_labels = [labels[i] for i in order]


    crack_lens = []
    crack_max_wids = []


    for lbl in sorted_labels:
        mask = img_labels == lbl


        median_axis, median_dist = morphology.medial_axis(mask, return_distance=True)

        crack_len = np.sum(median_axis)
        crack_max_wid = np.max(median_dist)
        crack_lens.append(crack_len)
        crack_max_wids.append(crack_max_wid)

    # Calculate crack metrics
    crack_length = np.sum(crack_lens)
    crack_max_width = np.max(crack_max_wids)
    crack_mean_width = np.sum(img_bnr) / crack_length if crack_length > 0 else 0

    # Convert the crack max width to millimeters
    crack_max_width_mm = crack_max_width
    print(crack_max_width_mm )

    # Crack severity classification based on max width in mm
    if crack_max_width_mm < 100: # 5 or 6
        severity = "low"
    elif 100 <= crack_max_width_mm <= 500:
        severity = "medium"
    else:
        severity = "high"

    # Move the image to the corresponding severity directory
    shutil.move(image_path, os.path.join(output_dirs[severity], image_name))
    print(f'{image_name} classified as {severity}.')

print("Images have been organized into 'low', 'medium', 'high', and 'no_crack' subdirectories based on crack severity.")


# In[ ]:





# In[13]:


pip install scikit-image


# In[4]:


import os
from PIL import Image

# Function to resize and rename images in a folder
def resize_and_rename_images(main_folder, subfolder, output_size=(224, 224)):
    """
    Resizes images in the given folder and renames them based on the main folder and subfolder name.

    Args:
    - main_folder (str): The name of the main folder (e.g., crack_500).
    - subfolder (str): The path to the subfolder (e.g., train, val, test).
    - output_size (tuple): Desired image size after resizing (default is 224x224).
    """
    # Get the name of the subfolder for renaming
    subfolder_name = os.path.basename(subfolder)

    # Create a list of valid image extensions
    valid_extensions = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.Jpeg')

    # List all files in the subfolder
    images = [f for f in os.listdir(subfolder) if f.endswith(valid_extensions)]

    # Process each image
    for idx, image_name in enumerate(images, 1):
        image_path = os.path.join(subfolder, image_name)
        try:
            # Open the image
            img = Image.open(image_path)

            # Resize the image
            img_resized = img.resize(output_size)

            # Create the new image name
            new_image_name = f"{main_folder}_{subfolder_name}_{idx}.png"

            # Define the new image path
            new_image_path = os.path.join(subfolder, new_image_name)

            # Save the resized and renamed image
            img_resized.save(new_image_path)

            # Optionally, delete the original image if you don't need it anymore
            os.remove(image_path)

            print(f"Processed {image_name} -> {new_image_name}")

        except Exception as e:
            print(f"Error processing {image_name}: {e}")


# In[5]:


main_folder = 'Crack'  # Name of the main dataset folder
subfolder = 'Dataset1_Pune/ML_DATASET - Copy/train'  # Change this to 'train', 'val', or 'test' based on the folder you're processing

# Call the function to resize and rename images in the 'train' subfolder
resize_and_rename_images(main_folder, subfolder)


# In[34]:


import os
import shutil

# Define source and destination base directories
png_base_dir = "Datasetnew_Pune/FinalFootpathdamagedataset/val"
jpg_base_dir = "Datasetnew_Pune/FinalFootpathdamagedataset/val"  # If JPGs are in the same folder as PNGs, use same path

dest_base_dir = "Dataset1_Pune/FinalFootpathdamagedataset/val"  # Should contain `low`, `medium`, `high` folders

# Go through each subfolder (low, medium, high)
for category in ['low', 'medium', 'high']:
    png_category_dir = os.path.join(dest_base_dir, category)
    jpg_dest_dir = os.path.join(dest_base_dir, category)

    for fname in os.listdir(png_category_dir):
        if fname.endswith(".png"):
            base_name = os.path.splitext(fname)[0]
            jpg_file = base_name + ".jpg"
            src_jpg_path = os.path.join(jpg_base_dir, jpg_file)
            dest_jpg_path = os.path.join(jpg_dest_dir, jpg_file)

            if os.path.exists(src_jpg_path):
                shutil.copy2(src_jpg_path, dest_jpg_path)
                print(f"Copied: {jpg_file} → {category}")
            else:
                print(f"JPG not found for: {jpg_file}")


# In[61]:


import os

# Set the base directory where 'low', 'medium', and 'high' subfolders are
base_dir = "Datasetnew_Pune/FinalFootpathdamagedataset/train"

# Initialize counters
jpg_count = 0
png_count = 0

# Loop through all subfolders (recursively) in the base directory
for root, dirs, files in os.walk(base_dir):
    # Loop through files in each subfolder
    for file in files:
        if file.lower().endswith('.jpg'):
            jpg_count += 1
        elif file.lower().endswith('.png'):
            png_count += 1

# Print results
print(f"Total .jpg images: {jpg_count}")
print(f"Total .png images: {png_count}")
print(f"Total images (jpg + png): {jpg_count + png_count}")


# In[36]:


import os
import shutil
import random

# Set paths for the test and train folders
test_folder = "Datasetnew_Pune/FinalFootpathdamagedataset/test"
train_folder = "Datasetnew_Pune/FinalFootpathdamagedataset/train"

# Get all PNG files in the test folder
png_files = [f for f in os.listdir(test_folder) if f.lower().endswith('.png')]

# Randomly select 30 PNG files
selected_pngs = random.sample(png_files, 30)

# Move selected images (both PNG and corresponding JPG)
for png_file in selected_pngs:
    # Move PNG image to the train folder
    png_src = os.path.join(test_folder, png_file)
    png_dest = os.path.join(train_folder, png_file)
    shutil.move(png_src, png_dest)

    # Find the corresponding JPG image
    jpg_file = os.path.splitext(png_file)[0] + ".jpg"
    jpg_src = os.path.join(test_folder, jpg_file)
    
    # Check if the corresponding JPG exists and move it
    if os.path.exists(jpg_src):
        jpg_dest = os.path.join(train_folder, jpg_file)
        shutil.move(jpg_src, jpg_dest)
        print(f"Moved {png_file} and {jpg_file} to the train folder.")
    else:
        print(f"JPG for {png_file} not found.")


# In[37]:


import os
import shutil

# Set source and destination folders
source_folder = "priti/Dataset1_Pune/FinalFootpathdamagedataset/train1"  # Folder where the .png files are stored
destination_folder = "priti/Datasetnew_Pune/FinalFootpathdamagedataset/train"  # Folder containing subfolders (low, medium, high)

# Define subfolders in the destination folder
subfolders = ['low', 'medium', 'high']

# Loop through each subfolder (low, medium, high)
for subfolder in subfolders:
    subfolder_path = os.path.join(destination_folder, subfolder)

    # Make sure the subfolder exists
    if not os.path.exists(subfolder_path):
        print(f"Subfolder {subfolder} does not exist.")
        continue

    # Loop through each file in the subfolder
    for file in os.listdir(subfolder_path):
        if file.lower().endswith('.png'):  # Look for .png files
            # Find the base name (without extension) of the png file
            base_name = os.path.splitext(file)[0]
            # Construct the corresponding png filename
            png_file = base_name + ".png"
            # Define the source path for the png file
            png_src = os.path.join(source_folder, png_file)
            # Define the destination path for the png file in the train folder
            png_dest = os.path.join(subfolder_path, png_file)

            # Check if the corresponding png file exists in the source folder
            if os.path.exists(png_src):
                # Copy the png file to the destination folder
                shutil.copy2(png_src, png_dest)
                print(f"Copied {png_file} to {subfolder_path}")
            else:
                print(f"PNG not found for {file}")


# In[82]:


import os
import shutil

# Set source and destination folders
source_folder = "/extra-Copy1"  # Folder containing subfolders (low, medium, high)
destination_folder = "priti/remainingcsv"  # Folder where you want to copy the PNG files

# Define subfolders in the source folder
subfolders = ['low', 'medium', 'high']

# Loop through each subfolder (low, medium, high) in the source folder
for subfolder in subfolders:
    subfolder_path = os.path.join(source_folder, subfolder)

    # Ensure the subfolder exists in the source folder
    if not os.path.exists(subfolder_path):
        print(f"Subfolder {subfolder} does not exist in source folder.")
        continue

    # Ensure the subfolder exists in the destination folder
    destination_subfolder = os.path.join(destination_folder, subfolder)
    if not os.path.exists(destination_subfolder):
        os.makedirs(destination_subfolder)  # Create the subfolder if it doesn't exist

    # Loop through each file in the subfolder
    for file in os.listdir(subfolder_path):
        if file.lower().endswith('.png'):  # Check for .png files
            # Construct the full path for the source .png file
            png_src = os.path.join(subfolder_path, file)
            # Construct the full path for the destination .png file
            png_dest = os.path.join(destination_subfolder, file)

            # Copy the .png file from the source to the destination subfolder
            shutil.copy2(png_src, png_dest)
            print(f"Copied {file} from {subfolder_path} to {destination_subfolder}")
import os
import shutil

# Set source and destination folders
source_folder = "priti/Dataset1_Pune/FinalFootpathdamagedataset/train1"
destination_folder = "priti/Datasetnew_Pune/FinalFootpathdamagedataset/train"

# Define subfolders in the source folder
subfolders = ['low', 'medium', 'high']

# Create destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Loop through each subfolder (low, medium, high)
for subfolder in subfolders:
    subfolder_path = os.path.join(source_folder, subfolder)

    # Check if the subfolder exists
    if not os.path.exists(subfolder_path):
        print(f"Subfolder {subfolder} does not exist in source folder.")
        continue

    # Loop through each file in the subfolder
    for file in os.listdir(subfolder_path):
        if file.lower().endswith('.png'):
            source_file = os.path.join(subfolder_path, file)
            destination_file = os.path.join(destination_folder, file)

            # If the destination file already exists, optionally rename to avoid overwrite
            if os.path.exists(destination_file):
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(destination_file):
                    destination_file = os.path.join(destination_folder, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.copy2(source_file, destination_file)
            print(f"Copied {file} from {subfolder_path} to {destination_folder}")


# In[43]:


import os
import shutil

# Set source and destination folders
source_folder = "Dataset1_Pune/FinalFootpathdamagedataset/val"  # Folder containing subfolders (low, medium, high)
destination_folder = "Datasetnew_Pune/FinalFootpathdamagedataset/val"  # Folder where you want to copy the PNG files

# Define subfolders in the source folder
subfolders = ['low', 'medium', 'high']

# Loop through each subfolder (low, medium, high) in the source folder
for subfolder in subfolders:
    subfolder_path = os.path.join(source_folder, subfolder)

    # Ensure the subfolder exists in the source folder
    if not os.path.exists(subfolder_path):
        print(f"Subfolder {subfolder} does not exist in source folder.")
        continue

    # Ensure the subfolder exists in the destination folder
    destination_subfolder = os.path.join(destination_folder, subfolder)
    if not os.path.exists(destination_subfolder):
        os.makedirs(destination_subfolder)  # Create the subfolder if it doesn't exist

    # Loop through each file in the subfolder
    for file in os.listdir(subfolder_path):
        if file.lower().endswith('.png'):  # Check for .png files
            # Construct the full path for the source .png file
            png_src = os.path.join(subfolder_path, file)
            # Construct the full path for the destination .png file
            png_dest = os.path.join(destination_subfolder, file)

            # Copy the .png file from the source to the destination subfolder
            shutil.copy2(png_src, png_dest)
            print(f"Copied {file} from {subfolder_path} to {destination_subfolder}")


# In[ ]:




