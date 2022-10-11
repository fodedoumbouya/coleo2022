import cv2

def filtre_gwe(data_list:list, accuracy:int = 5, min_area:int = 100, min_length:int = 5) -> list:
    result = []

    for el in data_list:
        peri = cv2.arcLength(el, True)
        approx = cv2.approxPolyDP(el, 0.01 * peri, True)
        area = cv2.contourArea(el)
        x,y,w,h = cv2.boundingRect(el)

        if len(approx) > min_length and w <= (h + accuracy) and w >= (h - accuracy) and area > accuracy:
            result.append(el)
    
    return result