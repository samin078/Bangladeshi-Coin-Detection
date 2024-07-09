import cv2
import numpy as np
import math

# Define minimal and maximal radius
min_r = 22
max_r = 38

coins = cv2.imread('images/test2.jpg', 1)

def edge_detect_coins():
    """
    Import the coins.jpg image and detect the edges of the coins.
    """
    coins_height, coins_width, coins_channel = coins.shape

    # Optimisation by decreasing the size of image, resulting in 4x faster run time
    coins_resized = cv2.resize(coins, (int(coins_width/2), int(coins_height/2)))

    # Blur to optimise edge finding
    coins_blurred = cv2.GaussianBlur(coins_resized, (5, 5), cv2.BORDER_DEFAULT)

    # Used Canny to find the edge
    coins_edge = cv2.Canny(coins_blurred, 127, 255)

    cv2.imwrite("coins_blurred.jpg", coins_blurred)
    cv2.imwrite("coins_edge.jpg", coins_edge)

    return coins_edge

def coin_center_detect():
    """
    Aim is to find the edges, find the radius of the coin and save the coordinates of the centers.
    """
    # Image with edges of coins detected
    coins_edge = edge_detect_coins()

    # Obtain the image size
    max_height, max_width = coins_edge.shape

    edge_threshold = 0.35  # How many pixels need to pass to be considered a coin edge
    intensity_threshold = 255 * 0.123  # The min value of pixel intensity to be considered edge
    next_circle_step = 1  # The amount of pixels to move to start comparing again
    coin_detection = []

    # Draw circles
    for radius in range(min_r, max_r):
        img_circle = np.zeros((radius * 2, radius * 2, 1), np.uint8)
        circle = cv2.circle(img_circle, (radius, radius), radius, 255)

        circumference = 2 * math.pi * radius
        circle_pixels = []

        for y in range(len(circle)):
            for x in range(len(circle[y])):
                if circle[x][y] == 255:
                    circle_pixels.append((x, y))

        print(('radius', radius))

        # Move circle through image
        for start_y in range(0, max_height - 2 * radius, next_circle_step):
            for start_x in range(0, max_width - 2 * radius, next_circle_step):
                count = 0

                # Cycle through the coordinates of circle
                for (x, y) in circle_pixels:
                    image_y = start_y + y
                    image_x = start_x + x

                    if coins_edge[image_y][image_x] >= intensity_threshold:
                        count += 1

                if count > 50:
                    percentage = round(count / circumference * 100, 2)
                    coor_x = start_x + radius
                    coor_y = start_y + radius
                    print(('candidate', coor_x, coor_y, radius, percentage))

                if (count / circumference) > edge_threshold:
                    coor_x = start_x + radius
                    coor_y = start_y + radius
                    coin_detection.append((coor_x, coor_y, radius))  # center
                    print(('-----------------', start_x + radius, start_y + radius, radius))

    return coin_detection

def circle_coins():
    coins_circled = coin_center_detect()
    coins_copy = coins.copy()
    for detected_circle in coins_circled:
        x_coor, y_coor, detected_radius = detected_circle
        coins_detected = cv2.circle(coins_copy, (x_coor*2, y_coor*2), detected_radius*2, (0, 0, 255), 1)

    cv2.imwrite("output.jpg", coins_detected)

def hough_circle_detection():
    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        img,  # Source image
        cv2.HOUGH_GRADIENT,  # Type of detection
        1,
        40,
        param1=50,
        param2=30,
        minRadius=min_r*2,  # Minimal radius
        maxRadius=max_r*2,  # Max radius
    )

    coins_copy = coins.copy()

    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        coins_detected = cv2.circle(coins_copy, (x_coor, y_coor), detected_radius, (0, 0, 255), 1)

    cv2.imwrite("coins_detected_Hough.jpg", coins_detected)

def compare_circle_detection():
    circle_coins()
    hough_circle_detection()
