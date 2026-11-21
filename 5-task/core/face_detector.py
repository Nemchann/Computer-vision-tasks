import cv2
import numpy as np

# Загружаем встроенный классификатор Хаара (идет в комплекте с OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(image: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5) -> list:
    """
    Обнаруживает лица на изображении с помощью каскадов Хаара.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detectMultiScale возвращает (x, y, w, h)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Приводим к формату (top, right, bottom, left) для совместимости
    face_locations = []
    for (x, y, w, h) in faces:
        face_locations.append((y, x + w, y + h, x))

    return face_locations


def draw_face_boxes(image: np.ndarray, face_locations: list,
                    color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Рисует прямоугольники вокруг обнаруженных лиц."""
    result_image = image.copy()
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(result_image, (left, top), (right, bottom), color, thickness)
    return result_image


def count_faces(image: np.ndarray) -> int:
    return len(detect_faces(image))


def validate_single_face(image: np.ndarray) -> tuple:
    """Проверяет, что на изображении ровно одно лицо."""
    face_locations = detect_faces(image)
    count = len(face_locations)

    if count == 0:
        return False, "На изображении не обнаружено лиц", None
    elif count == 1:
        return True, "Обнаружено одно лицо", face_locations[0]
    else:
        return False, f"Обнаружено {count} лиц. Требуется ровно одно лицо.", None


def get_face_crop(image: np.ndarray, face_location: tuple, padding: int = 20) -> np.ndarray:
    """Вырезает область лица с небольшим отступом."""
    top, right, bottom, left = face_location
    h, w = image.shape[:2]
    top = max(0, top - padding)
    bottom = min(h, bottom + padding)
    left = max(0, left - padding)
    right = min(w, right + padding)
    return image[top:bottom, left:right]