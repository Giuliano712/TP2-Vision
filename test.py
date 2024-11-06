import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data
from skimage.color import label2rgb
from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import expand_labels, watershed

img_colored = cv2.imread("Images/Echantillion1Mod2_301.png")
img_colored = cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)

img = apply_clahe(img_colored)
# Threshold the image
thresh = cv2.adaptiveThreshold(
    img, 175, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

edges = sobel(img)

markers = np.zeros_like(img)
foreground, background = 1, 2
markers[img < 30] = background
markers[img > 150] = foreground


ws = watershed(edges, markers)
seg = label(ws == foreground)
data = []

# Créer une copie pour dessiner les contours et numéros
img_contours = img_colored.copy()
img_contours_blurred = img_colored.copy()

# Trouver les contours de chaque segment
contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dessiner les contours et numéroter les régions
for i, contour in enumerate(contours):
    cv2.drawContours(img_contours, [contour], -1, (0, 255, 0), 2)  # Dessiner le contour en vert
    # Calculer le centre du contour pour positionner le numéro
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # Ajouter le numéro de l'étiquette
        cv2.putText(img_contours, str(i + 1), (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Numéro bleu
        
    mask = np.zeros(img_colored.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    b_mean = np.mean(img_colored[:, :, 0][mask == 255])
    g_mean = np.mean(img_colored[:, :, 1][mask == 255])
    r_mean = np.mean(img_colored[:, :, 2][mask == 255])

    data.append([i + 1, b_mean, g_mean, r_mean])

    df = pd.DataFrame(data, columns=["Numéro", "B_mean", "G_mean", "R_mean"])


# Afficher les résultats
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(15, 15),
    sharex=True,
    sharey=True,
)
print(df)

axes[0].imshow(img_colored, cmap="Greys_r")
axes[0].set_title("Original")

axes[1].imshow(img_contours)
axes[1].set_title("Contours et Numérotation")

for a in axes:
    a.axis("off")
fig.tight_layout()
plt.show()