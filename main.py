import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract


# --- Tesseract ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- MODELE YOLO ---
door_model = YOLO("door_detect.pt")
number_model = YOLO("number_detect.pt")

# --- WCZYTANIE OBRAZU ---
image_path = "dataset/images/test/20251120_131840.jpg"
img = cv2.imread(image_path)

def prepare_for_ocr_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # używamy V jako bazę
    base = v

    # lekki blur
    base = cv2.GaussianBlur(base, (3,3), 0)

    # binaryzacja Otsu
    _, th = cv2.threshold(base, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th


if img is None:
    print("Błąd: nie można otworzyć zdjęcia.")
    exit()

# --------------------------------------------------------
# 1️⃣ ROZCIĄGNIĘCIE HISTOGRAMU (Normalization)
# --------------------------------------------------------
# podnosi kontrast zdjęcia wejściowego
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# --------------------------------------------------------
# 2️⃣ PEŁNA KONWERSJA BGR → HSV (przykład użycia)
# --------------------------------------------------------
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# można użyć HSV do filtracji, ale tutaj tylko demonstracja

# --------------------------------------------------------
# 3️⃣ DETEKCJA DRZWI
# --------------------------------------------------------
door_results = door_model(img, conf=0.10)[0]
door_boxes = door_results.boxes.xyxy.cpu().numpy()

print(f"Znalezione drzwi: {len(door_boxes)}")

# --- RYSOWANIE BBOX DRZWI ---
for box in door_boxes:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, "Drzwi", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# --------------------------------------------------------
# 4️⃣ DETEKCJA NUMERU NA DRZWIACH + OCR
# --------------------------------------------------------

for i, box in enumerate(door_boxes):
    x1, y1, x2, y2 = map(int, box)

    # Wycięcie drzwi
    door_roi = img[y1:y2, x1:x2]
    cv2.imwrite(f"door_{i}.jpg", door_roi)

    # Detekcja numerów wewnątrz drzwi
    number_results = number_model(door_roi, conf=0.05)[0]
    number_boxes = number_results.boxes.xyxy.cpu().numpy()

    if len(number_boxes) == 0:
        print(f"Drzwi {i}: nie znaleziono numeru.")
        continue

    # Bierzemy pierwszy numer
    nx1, ny1, nx2, ny2 = map(int, number_boxes[0])
    number_roi = door_roi[ny1:ny2, nx1:nx2]
    cv2.imwrite(f"door_{i}_number.jpg", number_roi)

    # --- RYSOWANIE BBOX NUMERU ---
    cv2.rectangle(door_roi, (nx1, ny1), (nx2, ny2), (0,0,255), 2)
    cv2.putText(door_roi, "Numer", (nx1, ny1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # --------------------------------------------------------
    # 5️⃣ KOREKCJA PERSPEKTYWY NUMERU
    # --------------------------------------------------------
    # Upraszczamy: przyjmujemy prosty bounding box
    h, w = number_roi.shape[:2]

    src_points = np.float32([[0,0], [w,0], [0,h], [w,h]])
    dst_points = np.float32([[0,0], [200,0], [0,100], [200,100]])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected = cv2.warpPerspective(number_roi, M, (200, 100))

    # --------------------------------------------------------
    # 6️⃣ PRZYGOTOWANIE DO OCR
    # --------------------------------------------------------
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #th = prepare_for_ocr_hsv(corrected)
    # --------------------------------------------------------
    # 7️⃣ OCR
    # --------------------------------------------------------
    text = pytesseract.image_to_string(
        th, lang="eng+pol", config="--psm 6"
    ).strip()

    print(f"Drzwi {i} – odczytany numer: {text}")
    cv2.imwrite(f"door_{i}_number_for_ocr.jpg", th)

# --------------------------------------------------------
# 8️⃣ POKAZANIE OBRAZU Z BBOXAMI
# --------------------------------------------------------
cv2.imshow("Wynik detekcji", corrected)
cv2.imshow("Wynik detekcji 2", number_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()


