import cv2
import numpy as np

image = cv2.imread('testImage.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

colors = {
    'Red': ([0, 120, 70], [10, 255, 255]),
    'Green': ([40, 40, 40], [70, 255, 255]),
    'Blue': ([100, 150, 0], [140, 255, 255]),
    'Cyan': ([81, 100, 100], [93, 255, 255]),
    'Yellow': ([20, 100, 100], [30, 255, 255]),
    'Pink': ([140, 100, 100], [170, 255, 255]),
    'Orange': ([10, 100, 20], [25, 255, 255]),
    'Purple': ([130, 50, 50], [160, 255, 255])
}

for color_name, (lower, upper) in colors.items():
    lower_np = np.array(lower)
    upper_np = np.array(upper)
    mask = cv2.inRange(hsv, lower_np, upper_np)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f'{color_name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow('Multi Color Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()