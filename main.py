import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract

#ustawenie sciezki do OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#wczytanie modeli
door_model = YOLO("door_detect.pt") # link: https://hub.ultralytics.com/models/u53jjND2UndNgdnkjNnY
number_model = YOLO("number_detect.pt") #link: https://hub.ultralytics.com/models/u53jjND2UndNgdnkjNnY

# funkcje przygotywujace obraz do OCR

def apply_filters(img):

    #Median Blur
    #img = cv2.medianBlur(img, 3)

    #Bilateral Filter
    img = cv2.bilateralFilter(img, 9, 75, 75)

    #Gaussian Blur
    #img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def apply_morphology(binary):

    kernel = np.ones((5, 5), np.uint8)

    # Otwarcie (usuwa drobne białe piksele)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Zamknięcie (ułatwia OCR przy przerwanych literach)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closed


def prepare_for_ocr_hsv(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)

    v_min = np.min(v)
    v_max = np.max(v)

    if (v_max - v_min) > 10:
        v_norm = (v - v_min) * (255.0 / (v_max - v_min))
        v_norm = v_norm.astype(np.uint8)
    else:
        v_norm = v.copy()

    # Filtry
    v_filtered = apply_filters(v_norm)

    # Binaryzacja Otsu
    _, th = cv2.threshold(v_filtered, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morfologia: otwarcie + zamknięcie
    th = apply_morphology(th)

    return th


#wczytanie obrazu
image_path = "dataset/images/test/zd3.jpg"
img = cv2.imread(image_path)

#detekcja drzwi
door_results = door_model(img, conf=0.10)[0]
door_boxes = door_results.boxes.xyxy.cpu().numpy()

#zaznaczenie drzwi
for box in door_boxes:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)

#detekcja numeru na drzwiach
for i, box in enumerate(door_boxes):
    x1, y1, x2, y2 = map(int, box)
    door_roi = img[y1:y2, x1:x2]

    number_results = number_model(door_roi, conf=0.05)[0]
    number_boxes = number_results.boxes.xyxy.cpu().numpy()

    if len(number_boxes) == 0:
        print(f"Drzwi {i}: brak numeru.")
        continue

    # Pierwszy znaleziony numer
    nx1, ny1, nx2, ny2 = map(int, number_boxes[0])

    # Wycięcie numeru
    number_roi = door_roi[ny1:ny2, nx1:nx2]

    # Korekcja perspektywy
    h, w = number_roi.shape[:2]
    src = np.float32([[0,0],[w,0],[0,h],[w,h]])
    dst = np.float32([[0,0], [200,0], [0,100], [200,100]])
    M = cv2.getPerspectiveTransform(src, dst)

    corrected = cv2.warpPerspective(number_roi, M, (200,100))

    # Przygotowanie do OCR
    th = prepare_for_ocr_hsv(corrected)

    # OCR
    text = pytesseract.image_to_string(th, lang="eng", config="--psm 6 -c tessedit_char_whitelist=0123456789")
    print(f"Drzwi {i}: odczytany numer: {text.strip()}")
    #Rysowanie na orginalnym obrazie
    cv2.rectangle(img,(x1 + nx1, y1 + ny1),(x1 + nx2, y1 + ny2),(0, 0, 255), 3)

    #wyswietlenie obrazu numeru przygotowanego do OCR
    cv2.imshow("obraz do OCR", th)
    cv2.imshow("numer przed korekcja perspektywy", number_roi)
    cv2.imshow("numeru po korekcji perspektywy", corrected)

# Wyświetlenie
img = cv2.resize(img, (620, 480))
cv2.imshow("Detekcja", img)

cv2.waitKey(0)
cv2.destroyAllWindows()


