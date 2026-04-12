import cv2
import numpy as np
import os
import sys

def segment_image(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None: return
    
    # 1. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use Otsu's thresholding to handle uneven lighting
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Remove noise and connect nearby strokes
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 3. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours (Top to bottom, Left to right)
    boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 5 and h > 10: # Filter out small noise dots
            boxes.append((x, y, w, h))
    
    # Sort by Y (line) then X (character)
    boxes = sorted(boxes, key=lambda b: (b[1] // 50, b[0]))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (x, y, w, h) in enumerate(boxes):
        # Add a small padding
        char_img = img[max(0, y-2):y+h+2, max(0, x-2):x+w+2]
        cv2.imwrite(f"{output_dir}/char_{i:03d}.png", char_img)

if __name__ == "__main__":
    segment_image(sys.argv[1], sys.argv[2])
