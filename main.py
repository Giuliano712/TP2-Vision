import cv2
import matplotlib.pyplot as plt
import glob

from object_segmentation import ObjectSegmentation

# Render all images

def main():

    # Initialize figure for plotting
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing

    # Create the object segmentation class
    obj_seg = ObjectSegmentation()

    # Load the images from "Images" folder
    paths_objs = glob.glob('Images/Echantillion1Mod2_*.*')

    for idx, (ax, path_obj) in enumerate(zip(axes.flat, paths_objs)):
        # Load the image
        img_colored = cv2.imread(path_obj)

        # Segment the image
        segmented_image, df = obj_seg.segment(img_colored)

        # Resize from 600 600 to 300 300
        img_colored = cv2.resize(img_colored, (300, 300))

        row = idx // 3
        col = (idx % 3) * 2
        # Display original image
        axes[row, col].imshow(img_colored, extent=(0, 0.5, 0, 0.5))
        axes[row, col].axis('off')  # Hide axis for cleaner display
        # Display processed image
        axes[row, col + 1].imshow(segmented_image, extent=(0, 1, 0, 1))
        axes[row, col + 1].axis('off')  # Hide axis for cleaner display

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    

    # Display each image
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(images_rgb[i])
    #     ax.axis('off')  # Hide axis for cleaner display

    # Display the image
    # cv2.imshow("Segmented Image", segmented_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()