import cv2
import numpy as np
from core.face_detector import validate_single_face, get_face_crop

# Стандартный размер лица для алгоритма LBPH
TARGET_SIZE = (100, 100)


def extract_face_encoding(image: np.ndarray) -> np.ndarray:
    """
    Подготавливает изображение лица для алгоритма LBPH.
    Переводит в оттенки серого и изменяет размер до 100x100.
    """
    # Проверяем, что ровно одно лицо
    is_valid, message, face_location = validate_single_face(image)
    if not is_valid:
        raise ValueError(message)

    # Вырезаем лицо
    face_crop = get_face_crop(image, face_location, padding=10)

    # Переводим в оттенки серого (LBPH работает только с grayscale)
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # Приводим к единому размеру
    resized_face = cv2.resize(gray_face, TARGET_SIZE)

    return resized_face


def extract_single_face_encoding(image: np.ndarray) -> np.ndarray:
    """Совместимый интерфейс."""
    return extract_face_encoding(image)


def compare_encodings(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
    """
    Вычисляет разницу между двумя подготовленными изображениями лиц.
    Используется как метрика для эвристической оценки (чем меньше, тем лучше).
    """
    # Среднеквадратичная ошибка (MSE)
    mse = np.mean((encoding1.astype(float) - encoding2.astype(float)) ** 2)
    return mse