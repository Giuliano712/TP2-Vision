import os
import sys
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from object_recognition import ObjectRecognition, Room

# Get image path from argument
argv = sys.argv
image_path = None

if len(argv) > 1:
    image_path = argv[1]

if not image_path:
    print("Please provide the directory Images which contains all the rooms as an argument")
    sys.exit(1)

if not os.path.exists(image_path):
    print(f"folder not found: {image_path}")
    sys.exit(1)

print(f"Folder path: {image_path}")

# Define the number of columns (10 for objects + 1 for the reference image)
num_cols = 11

# Initialize figure for plotting
fig, axes = plt.subplots(nrows=len(Room), ncols=num_cols, figsize=(18, 10))

# Loop over each room
for i, room in enumerate(Room):
    # Get the directory for the current room
    directory = os.path.join(os.getcwd(), image_path, room.value)

    # Retrieve the reference image
    path_ref = glob.glob(os.path.join(directory, 'Reference.*'))
    if not path_ref:
        print(f"No reference image found in {directory}")
        continue
    image_ref = cv2.imread(path_ref[0])
    image_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display

    # Plot the reference image in the first column
    axes[i, 0].imshow(image_ref)
    axes[i, 0].set_title(f"{room.value} - Reference")
    axes[i, 0].axis('off')

    # Retrieve all object images
    paths_objs = glob.glob(os.path.join(directory, 'IMG_*.*'))

    # Limit to 7 object images for display (since we already used 1 column for the reference)
    paths_objs = paths_objs[:num_cols - 1]

    # Loop over each object image and process
    for j, path_obj in enumerate(paths_objs):
        image_obj = cv2.imread(path_obj)
        image_obj = cv2.cvtColor(image_obj, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display

        # Create ObjectRecognition object and detect objects
        obj_rec = ObjectRecognition(image_obj, image_ref, room)
        obj_rec.detect_objects()
        image_obj = obj_rec.image_obj

        # Display the object recognition result
        axes[i, j + 1].imshow(image_obj)  # j + 1 because column 0 is for reference
        if obj_rec.percent >= 40:
            axes[i, j + 1].set_title(f"{j + 1} - {obj_rec.percent:.2f}% DANGER", color='red')
        else:
            axes[i, j + 1].set_title(f"{j + 1} - {obj_rec.percent:.2f}%")
        axes[i, j + 1].axis('off')

    # Fill remaining columns with white images if there are less than 10 objects
    for j in range(len(paths_objs), num_cols - 1):
        # Create a white image as a placeholder
        white_image = np.ones_like(image_ref) * 255
        axes[i, j + 1].imshow(white_image)
        axes[i, j + 1].axis('off')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

