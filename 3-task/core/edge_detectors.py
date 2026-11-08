import cv2
import numpy as np


def apply_canny(image: np.ndarray, threshold1: int, threshold2: int) -> np.ndarray:
    """
    Применяет детектор границ Кэнни.

    Args:
        image: Входное изображение (grayscale)
        threshold1: Нижний порог гистерезиса
        threshold2: Верхний порог гистерезиса

    Returns:
        Бинарное изображение с выделенными границами (0 или 255)
    """
    # Canny автоматически применяет гауссово размытие 5x5 внутри
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges


def apply_roberts(image: np.ndarray) -> np.ndarray:
    """
    Применяет оператор Робертса для обнаружения границ.
    Использует два ядра 2x2 для вычисления градиента по диагоналям.

    Args:
        image: Входное изображение (grayscale)

    Returns:
        Изображение с выделенными границами (uint8, 0-255)
    """
    # Ядра Робертса 2x2
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # Вычисляем градиенты по двум направлениям
    gx = cv2.filter2D(image, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(image, cv2.CV_32F, kernel_y)

    # Вычисляем величину градиента
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Нормализуем к диапазону [0, 255]
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return magnitude_normalized