import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

# Constants
ANGLE_STEP = 30
THRESHOLD = 130
THRESHOLD_TYPE = cv2.THRESH_BINARY_INV
CLOSE_ITERATIONS = 5
TEMPLATE_SIZE = 100
CANDIDATES_MIN_AREA = 2200  # Adjust this value to filter out smaller contours
MATCH_METHOD = cv2.TM_SQDIFF_NORMED
TEMPLATE_IMG = 'data/5-five-o-f.jpg'  # Default template image path
LABELED_IMAGE = 'labeled_image.bmp'  # Default output labeled image path
LABEL = 'Coin'  # Default label for coins
MATCH_THRESHOLD = 0.3  # Default match threshold

# Helper functions
def load_image(name, grayscale=False):
    image = cv2.imread(name, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if image is None or (grayscale and len(image.shape) != 2) or (not grayscale and image.shape[2] != 3):
        print(f"{name} could not be read or is not correct.")
        exit(1)
    return image

def create_mask(image):
    _, mask = cv2.threshold(image, THRESHOLD, 255, THRESHOLD_TYPE)
    composite_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None, iterations=CLOSE_ITERATIONS)
    return composite_mask

def locate(mask):
    moments = cv2.moments(mask, True)
    area = moments['m00']
    if area == 0:
        raise ValueError("No non-zero pixels found in mask. Check the threshold values or input image.")
    radius = np.sqrt(area / np.pi)
    x_centroid = moments['m10'] / moments['m00']
    y_centroid = moments['m01'] / moments['m00']
    return np.array([[x_centroid, y_centroid, radius]], dtype=np.float32)

def center_and_scale(image, mask, characteristics):
    radius = characteristics[0, 2]
    x_center = characteristics[0, 0]
    y_center = characteristics[0, 1]
    diameter = int(round(radius * 2))
    
    x_org = max(0, int(round(x_center - radius)))
    y_org = max(0, int(round(y_center - radius)))

    x_end = min(image.shape[1], x_org + diameter)
    y_end = min(image.shape[0], y_org + diameter)

    if x_end - x_org <= 0 or y_end - y_org <= 0:
        print("ROI exceeds image dimensions after adjustment. Skipping this candidate.")
        return None

    roi_img = image[y_org:y_end, x_org:x_end]
    roi_mask = mask[y_org:y_end, x_org:x_end]
    
    # Create a circular mask for the ROI
    circle_mask = np.zeros((y_end - y_org, x_end - x_org), dtype=np.uint8)
    cv2.circle(circle_mask, ((x_end - x_org) // 2, (y_end - y_org) // 2), (x_end - x_org) // 2, (255), thickness=cv2.FILLED)
    circular_roi_img = cv2.bitwise_and(roi_img, roi_img, mask=circle_mask)

    if circular_roi_img.shape[0] != TEMPLATE_SIZE or circular_roi_img.shape[1] != TEMPLATE_SIZE:
        circular_roi_resized = cv2.resize(circular_roi_img, (TEMPLATE_SIZE, TEMPLATE_SIZE))
    else:
        circular_roi_resized = circular_roi_img

    return circular_roi_resized

def save_rotated_templates(image, step_angle):
    if not os.path.exists('templates'):
        os.makedirs('templates')
    for angle in range(0, 360, step_angle):
        center = (TEMPLATE_SIZE // 2, TEMPLATE_SIZE // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (TEMPLATE_SIZE, TEMPLATE_SIZE))
        cv2.imwrite(f'templates/template-{angle:03d}.bmp', rotated_image)
        # Debug information to ensure the templates are created
        print(f"Saved template-{angle:03d}.bmp")

def load_templates(angle_step):
    templates = []
    for angle in range(0, 360, angle_step):
        template_img = cv2.imread(f'templates/template-{angle:03d}.bmp', cv2.IMREAD_GRAYSCALE)
        if template_img is None:
            print(f"Could not read template-{angle:03d}.bmp")
            exit(1)
        templates.append(template_img)
    return templates

class Candidate:
    def __init__(self, image, x, y, radius, score):
        self.image = image
        self.x = x
        self.y = y
        self.radius = radius
        self.score = score

def get_candidates(image, mask):
    candidates = []
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        drawing = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(drawing, [contour], -1, (255), thickness=cv2.FILLED)
        moments = cv2.moments(drawing, True)
        area = moments['m00']
        if area < CANDIDATES_MIN_AREA:
            continue
        radius = np.sqrt(area / np.pi)
        x_centroid = moments['m10'] / moments['m00']
        y_centroid = moments['m01'] / moments['m00']
        characteristics = np.array([[x_centroid, y_centroid, radius]], dtype=np.float32)
        image_cs = center_and_scale(image, drawing, characteristics)
        if image_cs is not None:
            candidate = Candidate(image_cs, x_centroid, y_centroid, radius, 0)
            candidates.append(candidate)
    return candidates

def save_candidates(candidates):
    if not os.path.exists('candidates'):
        os.makedirs('candidates')
    for i, candidate in enumerate(candidates):
        cv2.imwrite(f'candidates/Candidate-{i:03d}.bmp', candidate.image)

def match_candidates(templates, candidates):
    for candidate in candidates:
        match_candidate(templates, candidate)

def match_candidate(templates, candidate):
    if MATCH_METHOD in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        candidate.score = float('inf')
    else:
        candidate.score = 0
    for template_img in templates:
        result = cv2.matchTemplate(candidate.image, template_img, MATCH_METHOD)
        score = result[0][0]
        if MATCH_METHOD in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            candidate.score = min(candidate.score, score)
        else:
            candidate.score = max(candidate.score, score)

def selected(candidate, threshold):
    if MATCH_METHOD in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        return candidate.score <= threshold
    else:
        return candidate.score > threshold

def draw_label(image, candidate, label):
    x = int(candidate.x - candidate.radius)
    y = int(candidate.y)
    point = (x, y)
    cv2.putText(image, label, point, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 128, 128), 2)

def label_coins(image, candidates, threshold, label):
    image_labeled = image.copy()
    for candidate in candidates:
        if selected(candidate, threshold):
            draw_label(image_labeled, candidate, label)
    return image_labeled

def main():
    # GUI to get user input image
    root = tk.Tk()
    root.withdraw()

    INPUT_IMAGE = filedialog.askopenfilename(title="Select Input Image")

    # Creating templates
    image = load_image(TEMPLATE_IMG, grayscale=True)
    mask = create_mask(image)
    try:
        loc = locate(mask)
    except ValueError as e:
        print(e)
        return
    image_cs = center_and_scale(image, mask, loc)
    save_rotated_templates(image_cs, ANGLE_STEP)

    # Loading templates and processing the input image
    templates = load_templates(ANGLE_STEP)
    image = load_image(INPUT_IMAGE, grayscale=True)
    image_copy = image.copy()

    # Detecting edges and applying morphological operations
    edges = cv2.Canny(image, 60, 150)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    morph = cv2.dilate(morph, kernel, iterations=1)
    morph = cv2.erode(morph, kernel, iterations=1)

    candidates = get_candidates(image, morph)
    save_candidates(candidates)
    match_candidates(templates, candidates)
    for candidate in candidates:
        print(candidate.score)
    labeled_img = label_coins(image_copy, candidates, MATCH_THRESHOLD, LABEL)
    cv2.imwrite(LABELED_IMAGE, labeled_img)

    # Display the results
    cv2.imshow("Edges", edges)
    cv2.imshow("Morphological Operations", morph)
    cv2.imshow("Contours", labeled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
