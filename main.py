import os
import sys

import cv2
import matplotlib.pyplot as plt
import glob

from object_segmentation import ObjectSegmentation


# Render all images

def main(path):
    # Initialize figure for plotting
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(12, 10),
                             gridspec_kw=dict(width_ratios=[1, 0.5, 1, 0.5, 1, 0.5]))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing

    # Create the object segmentation class
    obj_seg = ObjectSegmentation()

    # Load the images from path folder
    paths_objs = glob.glob(f"{path}/*.png")

    # Folder path
    folder_path = 'dataframes'

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    for idx, (ax, path_obj) in enumerate(zip(axes.flat, paths_objs)):
        # Get the file name
        file_name = os.path.splitext(os.path.basename(path_obj))[0]

        # Load the image
        img_colored = cv2.imread(path_obj)

        # Segment the image
        segmented_image, df = obj_seg.segment(img_colored)

        # Save the dataframe to a CSV file
        df.to_csv(f"{folder_path}/{file_name}.csv", index=False)

        row = idx // 3
        col = (idx % 3) * 2

        # Display processed image
        axes[row, col].imshow(segmented_image, vmin=0, vmax=1)
        axes[row, col].set_title(file_name)
        axes[row, col].axis('off')  # Hide axis for cleaner display

        # Display original image
        axes[row, col + 1].imshow(img_colored, vmin=0, vmax=1)
        axes[row, col + 1].axis('off')  # Hide axis for cleaner display

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


# Get image path from argument
argv = sys.argv
image_path = None

if len(argv) > 1:
    image_path = argv[1]

if not image_path:
    print("Please provide an image path as an argument")
    sys.exit(1)

if not os.path.exists(image_path):
    print(f"Image not found: {image_path}")
    sys.exit(1)

print(f"Image path: {image_path}")


if __name__ == "__main__":
    main(image_path)