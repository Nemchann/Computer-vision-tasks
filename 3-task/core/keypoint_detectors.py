import cv2
import numpy as np


def detect_sift(image: np.ndarray) -> tuple:
    """
    Выделяет ключевые точки и дескрипторы методом SIFT.
    SIFT инвариантен к масштабу и повороту изображения.

    Args:
        image: Входное изображение (grayscale)

    Returns:
        (result_image, keypoints, descriptors)
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # Рисуем ключевые точки с ориентацией и размером
    result = cv2.drawKeypoints(
        image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)  # Зелёный цвет кружков
    )

    return result, keypoints, descriptors


def detect_surf(image: np.ndarray) -> tuple:
    """
    Выделяет ключевые точки и дескрипторы методом SURF.
    SURF — ускоренный аналог SIFT, использует интегральные изображения.

    ВАЖНО: SURF запатентован. В OpenCV 4.x доступен только если библиотека
    собрана с флагом OPENCV_ENABLE_NONFREE. В противном случае возникнет ошибка.

    Args:
        image: Входное изображение (grayscale)

    Returns:
        (result_image, keypoints, descriptors)
    """
    # hessianThreshold = 400 — порог детектирования (чем больше, тем меньше точек)
    surf = cv2.SURF_create(hessianThreshold=400)
    keypoints, descriptors = surf.detectAndCompute(image, None)

    result = cv2.drawKeypoints(
        image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(255, 0, 0)  # Синий цвет кружков
    )

    return result, keypoints, descriptors


def detect_fast(image: np.ndarray, threshold: int = 20) -> tuple:
    """
    Выделяет ключевые точки методом FAST (Features from Accelerated Segment Test).
    FAST — очень быстрый детектор, но не создаёт дескрипторы и не инвариантен к масштабу.

    Args:
        image: Входное изображение (grayscale)
        threshold: Порог чувствительности (10-50, чем меньше — тем больше точек)

    Returns:
        (result_image, keypoints, None) — дескрипторы отсутствуют
    """
    fast = cv2.FastFeatureDetector_create(threshold=threshold)
    keypoints = fast.detect(image, None)

    # FAST не вычисляет дескрипторы, поэтому передаём None
    result = cv2.drawKeypoints(
        image, keypoints, None,
        color=(255, 0, 255)  # Пурпурный цвет кружков
    )

    return result, keypoints, None