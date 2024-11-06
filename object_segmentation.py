import os

import cv2
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.measure import label
import numpy as np
import pandas as pd

from utils import apply_clahe

class ObjectSegmentation:
    def __init__(self):
        pass
        
    def segment(self, bgr_image):

        # Convert the image from BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Apply CLAHE to enhance the contrast
        image = apply_clahe(rgb_image)

        # Threshold the image
        edges = sobel(image)

        # Add markers
        markers = np.zeros_like(image)
        foreground, background = 1, 2
        markers[image < 30] = background
        markers[image > 150] = foreground

        # Apply watershed algorithm
        ws = watershed(edges, markers)
        seg = label(ws == foreground)

        segmented_image = self.draw_image(seg, rgb_image)

        df = self.get_dataframes(seg, rgb_image)

        return segmented_image, df
    
    # Draw borders around the objects
    def draw_image(self, seg, rgb_image):

        # Create a copy to draw the contours and numbers
        img_contours = rgb_image.copy()
        # Trouver les contours de chaque segment
        contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dessiner les contours et numéroter les régions
        for i, contour in enumerate(contours):

            # Dessiner le contour
            cv2.drawContours(img_contours, [contour], -1, (0, 255, 0), 2)  # Dessiner le contour en vert

            # Calculer le centre du contour pour positionner le numéro
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Ajouter le numéro de l'étiquette
                cv2.putText(img_contours, str(i + 1), (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Numéro bleu

        return img_contours
    
    def get_dataframes(self, seg, rgb_image):

        data = []

        # Trouver les contours de chaque segment
        contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dessiner les contours et numéroter les régions
        for i, contour in enumerate(contours):

            mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            b_mean = np.mean(rgb_image[:, :, 0][mask == 255])
            g_mean = np.mean(rgb_image[:, :, 1][mask == 255])
            r_mean = np.mean(rgb_image[:, :, 2][mask == 255])

            data.append([i + 1, b_mean, g_mean, r_mean])

        # Create a dataframe from data
        df = pd.DataFrame(data, columns=["Numéro", "B_mean", "G_mean", "R_mean"])

        return df