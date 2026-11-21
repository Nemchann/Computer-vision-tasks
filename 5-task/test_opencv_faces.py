import cv2
from core.face_detector import detect_faces, draw_face_boxes, validate_single_face
from core.face_encoder import extract_face_encoding

# Замени на путь к твоему фото
image = cv2.imread("resources/test_faces/test_image1.JPG")

if image is None:
    print("Не удалось загрузить изображение")
    exit()

# 1. Детекция
face_locations = detect_faces(image)
print(f"Обнаружено лиц: {len(face_locations)}")

if len(face_locations) > 0:
    result = draw_face_boxes(image, face_locations, color=(0, 255, 0), thickness=3)
    cv2.imshow("OpenCV Face Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2. Подготовка признаков
    encoding = extract_face_encoding(image)
    print(f"Признаки подготовлены. Размер: {encoding.shape}")
    print("OpenCV работает стабильно! Никаких DLL и импортов.")