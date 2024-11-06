import cv2

from object_segmentation import ObjectSegmentation

# Render one image

def main():
    # Load the image
    img_colored = cv2.imread("Images/Echantillion1Mod2_301.png")

    # Create the object segmentation class
    obj_seg = ObjectSegmentation()

    # Segment the image
    segmented_image, df = obj_seg.segment(img_colored)

    # Display the image
    cv2.imshow("Segmented Image", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()