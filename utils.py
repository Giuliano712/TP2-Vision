import cv2

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)