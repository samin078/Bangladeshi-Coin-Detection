import math
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import shutil 
from tkinter import filedialog, Label, Button
from tkinter import ttk
from PIL import Image, ImageTk

# Constants
TEMPLATE_FOLDER = 'data'
ANGLE_STEP = 30
THRESHOLD = 130
THRESHOLD_TYPE = cv2.THRESH_BINARY_INV
CLOSE_ITERATIONS = 5
TEMPLATE_SIZE = 100
CANDIDATES_MIN_AREA = 2000  
MATCH_METHOD = cv2.TM_SQDIFF_NORMED
LABELED_IMAGE = 'labeled_image.bmp'  
MATCH_THRESHOLD = 0.135  

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
    # cv2.imshow("Mask", composite_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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

    # cv2.imshow("Circular ROI", circular_roi_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return circular_roi_resized

def load_templates_from_folder(folder_path, angle_step):
    templates = {}
    for filename in os.listdir(folder_path):
        if filename.startswith(('1', '2', '5')) and filename.endswith('.jpg'):
            value = filename.split('-')[0]
            image = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                if value not in templates:
                    templates[value] = []
                mask = create_mask(image)
                try:
                    loc = locate(mask)
                except ValueError as e:
                    print(e)
                    continue
                image_cs = center_and_scale(image, mask, loc)
                if image_cs is not None:
                    save_rotated_templates(image_cs, angle_step, templates[value])
    return templates


def save_rotated_templates(image, step_angle, templates):
    num_angles = 360 // step_angle
    grid_size = math.ceil(math.sqrt(num_angles))  

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    for i, angle in enumerate(range(0, 360, step_angle)):
        center = (TEMPLATE_SIZE // 2, TEMPLATE_SIZE // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (TEMPLATE_SIZE, TEMPLATE_SIZE))
        templates.append(rotated_image)

        row = i // grid_size
        col = i % grid_size
        axs[row, col].imshow(rotated_image, cmap='gray')
        axs[row, col].set_title(f'{angle}°')
        axs[row, col].axis('off')

    for j in range(i + 1, grid_size * grid_size):
        row = j // grid_size
        col = j % grid_size
        axs[row, col].axis('off')
        

    # plt.tight_layout()
    # plt.show()

class Candidate:
    def __init__(self, image, x, y, radius, scores, contour):
        self.image = image
        self.x = x
        self.y = y
        self.radius = radius
        self.scores = scores
        self.contour = contour

def get_candidates(image, mask):
    candidates = []
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    
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
        bounding_box = cv2.boundingRect(contour)
        bounding_boxes.append((bounding_box, area, contour))
    
    bounding_boxes.sort(key=lambda x: x[1], reverse=True)
    
    filtered_contours = []
    for i, (bbox1, area1, contour1) in enumerate(bounding_boxes):
        keep = True
        x1, y1, w1, h1 = bbox1
        for bbox2, area2, contour2 in bounding_boxes[:i]:
            x2, y2, w2, h2 = bbox2
            if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                keep = False
                break
        if keep:
            filtered_contours.append(contour1)
    
    for contour in filtered_contours:
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
        # cv2.imshow("Candidate", image_cs)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if image_cs is not None:
            candidate = Candidate(image_cs, x_centroid, y_centroid, radius, {}, contour)
            candidates.append(candidate)
            
    
    return candidates

def save_candidates(candidates):
    if os.path.exists('candidates'):
        shutil.rmtree('candidates')
    os.makedirs('candidates')
    for i, candidate in enumerate(candidates):
        cv2.imwrite(f'candidates/Candidate-{i:03d}.bmp', candidate.image)

def match_candidates(templates, candidates):
    for candidate in candidates:
        match_candidate(templates, candidate)

def match_candidate(templates, candidate):
    for value, template_list in templates.items():
        min_score = float('inf')
        for template_img in template_list:
            result = cv2.matchTemplate(candidate.image, template_img, MATCH_METHOD)
            score = result[0][0]
            min_score = min(min_score, score)
        candidate.scores[value] = min_score

def select_best_match(candidate):
    min_score_value = min(candidate.scores.values())
    best_matches = [value for value, score in candidate.scores.items() if score == min_score_value]
    return best_matches

def draw_label(image, candidate, label):
    x = int(candidate.x - candidate.radius)
    y = int(candidate.y)
    point = (x, y)
    cv2.putText(image, label, point, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

def label_coins(image, candidates):
    image_labeled = image.copy()
    final_labels = []

    for candidate in candidates:
        best_matches = select_best_match(candidate)
        final_labels.append(best_matches)
    
    coin_counts = {}
    
    for i, candidate in enumerate(candidates):
        best_label = max(set(final_labels[i]), key=final_labels[i].count)
        draw_label(image_labeled, candidate, best_label)
        cv2.drawContours(image_labeled, [candidate.contour], -1, (0, 255, 0), 2)
        if best_label in coin_counts:
            coin_counts[best_label] += 1
        else:
            coin_counts[best_label] = 1
    
    return image_labeled, coin_counts

def print_scores(candidates):
    header = "Candidate | " + " | ".join(sorted(candidates[0].scores.keys()))
    print(header)
    print("-" * len(header))
    for i, candidate in enumerate(candidates):
        scores = " | ".join(f"{candidate.scores[value]:.4f}" for value in sorted(candidate.scores.keys()))
        print(f"   {i:03d}    | " + scores)

def preprocess_image(image):
    edges = cv2.Canny(image, 60, 150)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    # Display the images using matplotlib
    titles = ['Original Image', 'Canny Edges', 'Morphological Transform']
    images = [image, edges, morph]
    
    plt.figure(figsize=(12, 4))
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    # plt.tight_layout()
    plt.show()

    return edges, morph

class CoinDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Coin Detection")
        self.input_image = None
        self.output_image = None

        self.load_button = Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.canvas_input = tk.Canvas(root, width=400, height=400)
        self.canvas_input.pack(side="left", padx=10, pady=10)

        self.canvas_output = tk.Canvas(root, width=400, height=400)
        self.canvas_output.pack(side="right", padx=10, pady=10)

        self.result_label = Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

    def load_image(self):
        image_path = filedialog.askopenfilename(title="Select Input Image")
        if not image_path:
            return

        self.input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.display_image(self.input_image, self.canvas_input)

        templates = load_templates_from_folder(TEMPLATE_FOLDER, ANGLE_STEP)
        edges,  morph = preprocess_image(self.input_image)
        candidates = get_candidates(self.input_image, morph)
        save_candidates(candidates)
        match_candidates(templates, candidates)

        labeled_img, coin_counts = label_coins(self.input_image.copy(), candidates)
        self.display_image(labeled_img, self.canvas_output)

        self.update_results(coin_counts)

    def display_image(self, image, canvas):
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        img_pil.thumbnail((400, 400), Image.Resampling.LANCZOS)  
        img_tk = ImageTk.PhotoImage(img_pil)
        canvas.create_image(200, 200, image=img_tk)
        canvas.image = img_tk


    def update_results(self, coin_counts):
        result_text = f"Total Coins: {sum(coin_counts.values())}\n"
        result_text += "\n".join(f"{coin}: {count}" for coin, count in coin_counts.items())
        self.result_label.config(text=result_text)


def main():
    root = tk.Tk()
    app = CoinDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
    