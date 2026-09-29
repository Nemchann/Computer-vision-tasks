import numpy as np
import cv2


def get_sharpen_kernel() -> np.ndarray:
    """
    Ядро для повышения резкости (усиление контуров).
    Центральное значение положительное, соседние — отрицательные.
    Сумма элементов = 1 (сохраняет общую яркость).
    """
    return np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)


def get_motion_blur_kernel(size: int, angle: float) -> np.ndarray:
    """
    Ядро для эффекта "размытие в движении".

    Args:
        size: Длина ядра (количество пикселей)
        angle: Угол размытия в градусах (0 = горизонтально, 90 = вертикально)

    Returns:
        numpy-массив с ядром размытия
    """
    # Создаём ядро в виде линии под заданным углом
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    # Вычисляем конечные точки линии
    angle_rad = np.deg2rad(angle)
    dx = int(np.cos(angle_rad) * center)
    dy = int(np.sin(angle_rad) * center)

    # Рисуем линию в ядре
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, thickness=1)

    # Нормализуем ядро (сумма элементов = 1)
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel = kernel / kernel_sum

    return kernel


def get_emboss_kernel() -> np.ndarray:
    """
    Ядро для эффекта "тиснение"
    Создаёт эффект объёмности за счёт выделения градиентов яркости.
    """
    return np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ], dtype=np.float32)


def apply_median_filter(image: np.ndarray, size: int) -> np.ndarray:
    """
    Применяет медианный фильтр к изображению.

    Args:
        image: Входное изображение (grayscale или цветное)
        size: Размер ядра (должен быть нечётным числом: 3, 5, 7, ...)

    Returns:
        Отфильтрованное изображение
    """
    # Размер должен быть нечётным
    if size % 2 == 0:
        size += 1

    return cv2.medianBlur(image, size)


def apply_custom_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Применяет пользовательский фильтр (свёртку) к изображению.

    Args:
        image: Входное изображение
        kernel: Ядро свёртки (может содержать отрицательные значения)

    Returns:
        Отфильтрованное изображение
    """
    # cv2.filter2D выполняет свёртку изображения с ядром
    result = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    return result