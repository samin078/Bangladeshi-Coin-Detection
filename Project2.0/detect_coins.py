import cv2
import numpy as np
from matplotlib import pyplot as plt

def preprocess_template(template_path):
    # Load the template image and convert to grayscale
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(template, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour which should correspond to the coin
    contour = max(contours, key=cv2.contourArea)
    
    return contour

def detect_coins(imagePath, template1, template2):
    # Load the image and convert to grayscale
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the input image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Define a threshold for contour matching
    threshold = 0.2
    
    # Draw rectangles around matched regions
    for contour in contours:
        match1 = cv2.matchShapes(contour, template1, 1, 0.0)
        match2 = cv2.matchShapes(contour, template2, 1, 0.0)
        
        if match1 < threshold:
            color = (0, 255, 0)  # Green for matched regions with template1
        elif match2 < threshold:
            color = (255, 0, 0)  # Blue for matched regions with template2
        else:
            color = (0, 0, 255)  # Red for non-matched regions
        
        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw the rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    # Display the result
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title('Detected Coins')
    plt.show()
