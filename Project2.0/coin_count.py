import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    return edges

def find_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def fit_circles(contours, image):
    circles = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) > 5:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:
                circles.append((int(x), int(y), int(radius)))
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
    return circles

def identify_coins(circles):
    koruny = {
        "1 CZK": {"value": 1, "radius": 20, "count": 0},
        "2 CZK": {"value": 2, "radius": 21.5, "count": 0},
        "5 CZK": {"value": 5, "radius": 23, "count": 0},
        "10 CZK": {"value": 10, "radius": 24.5, "count": 0},
        "20 CZK": {"value": 20, "radius": 26, "count": 0},
        "50 CZK": {"value": 50, "radius": 27.5, "count": 0},
    }

    tolerance = 1.5

    total_amount = 0

    for circle in circles:
        x, y, radius = circle
        for koruna, details in koruny.items():
            if abs(radius - details["radius"]) <= tolerance:
                details["count"] += 1
                total_amount += details["value"]
                break

    for koruna, details in koruny.items():
        print(f"{koruna}: {details['count']} pieces")

    print(f"Total amount: {total_amount} CZK")
    return total_amount

def main(image_path):
    edges = preprocess_image(image_path)
    cv2.imwrite('preprocessed_edges.jpg', edges)
    
    contours = find_contours(edges)
    coins_image = cv2.imread(image_path)
    circles = fit_circles(contours, coins_image)
    cv2.imwrite('detected_circles.jpg', coins_image)
    
    identify_coins(circles)

if __name__ == "__main__":
    main('images/test2.jpg')
