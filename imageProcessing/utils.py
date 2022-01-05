import cv2
# import pytesseract
import numpy as np


def detect_text(image):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Remove grey and black from Image
    lower_val = np.array([0,0,0])
    upper_val = np.array([255,0,255])
    mask = cv2.inRange(hsv_image, lower_val, upper_val)
    color_image = image.copy()
    color_image[np.where(mask)] = 255

    # Convert color of small spaces between two contours to white
    canny = cv2.Canny(color_image, 100, 200)
    cnts, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.001 * peri, True)
        x,y,w,h = cv2.boundingRect(approx)
        area = cv2.contourArea(c)
        aspect_ratio = max(w, h)/min(w, h)
        if area < 5:
            cv2.fillPoly(color_image, pts=[c], color=0)
            continue
        if aspect_ratio <1.2:
            cv2.fillPoly(color_image, pts=[c], color=0)
            continue

    # Find contours for the converted image
    canny = cv2.Canny(color_image, 100, 200)
    cnts, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    text_array = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.001 * peri, True)
        x,y,w,h = cv2.boundingRect(approx)
        area = cv2.contourArea(c)
        aspect_ratio = max(w, h)/min(w, h)
        if 0 < h < 20 and area > 5:
            croppedImage = image[y:y+ h, x:x+w]
            img = cv2.resize(croppedImage, (0, 0), fx=5, fy=5)
            # cv2.imwrite(f'large{i}.png', img)
            # ctext = pytesseract.image_to_string(croppedImage, lang='eng')
            i += 1
            # if len(ctext) > 0:
            text_array.append({'contour':c, 'text':""})
    return text_array

def detect_marked(image):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Remove grey and black from Image
    lower_val = np.array([0,0,0])
    upper_val = np.array([255,15,255])
    mask = cv2.inRange(hsv_image, lower_val, upper_val)
    color_image = image.copy()
    color_image[np.where(mask)] = 255

    # Blur image to remove noise
    kernel_size = 7
    blur_image = cv2.GaussianBlur(color_image,(kernel_size, kernel_size), 0)

    # Find the contour of larger areas
    canny = cv2.Canny(blur_image, 100, 200)
    kernel = np.array((1,1))
    imgClose = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    cnts, hierarchy = cv2.findContours(imgClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.001 * peri, True)
        x,y,w,h = cv2.boundingRect(approx)
        area = cv2.contourArea(c)
        if len(approx) > 50:
            cv2.rectangle(image, (x, y), (x + w, y + h), (100, 0, 0), 2)
    return cnts

def find_nearest(text_data, image_data):
    result = []
    for dict in text_data:
        text_contour = dict['contour']
        minimum = float('inf')
        c = None
        for i in range(text_contour.shape[0]):
            for cnt in image_data:
                for j in range(cnt.shape[0]):
                    dist = np.linalg.norm(text_contour[i]-cnt[j])
                    if dist < minimum:
                        c = cnt
        result.append({'text':dict['text'], 'contour':c})
    return result
