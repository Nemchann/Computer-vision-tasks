import cv2
import numpy as np


def detect_license_plate(image: np.ndarray, draw_rect: bool = True) -> tuple:
    """
    Выделяет автомобильный номер на изображении.
    Использует несколько методов для повышения надёжности.
    """
    result_image = image.copy()

    # Метод 1: Попробуем найти номер через контуры
    plate_rect = _detect_plate_by_contours(image)

    # Метод 2: Если не нашли, попробуем через морфологию
    if plate_rect is None:
        plate_rect = _detect_plate_by_morphology(image)

    # Если номер найден
    if plate_rect is not None:
        x, y, w, h = plate_rect

        if draw_rect:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            return result_image, True
        else:
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            plate_crop = image[y1:y2, x1:x2]
            return plate_crop, True
    else:
        return result_image, False


def _detect_plate_by_contours(image: np.ndarray) -> tuple:
    """Поиск номера через анализ контуров."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применяем размытие
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Детектор границ Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Морфологические операции для соединения разрывов
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Находим контуры
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_plate = None
    max_score = 0

    for contour in contours:
        # Аппроксимация
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = w * h
            image_area = image.shape[0] * image.shape[1]

            # Более гибкие проверки
            if 2.5 < aspect_ratio < 6.0 and 1000 < area < image_area * 0.5:
                contour_area = cv2.contourArea(contour)
                rect_area = w * h
                if rect_area > 0:
                    fill_ratio = contour_area / rect_area

                    # Score: чем ближе к идеальным пропорциям, тем лучше
                    aspect_score = 1.0 - abs(aspect_ratio - 3.5) / 3.0
                    score = aspect_score * (0.5 + fill_ratio)

                    if score > max_score:
                        max_score = score
                        best_plate = (x, y, w, h)

    return best_plate


def _detect_plate_by_morphology(image: np.ndarray) -> tuple:
    """
    Поиск номера через морфологические операции.
    Хорошо работает для горизонтальных прямоугольных объектов.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Бинаризация (Otsu's thresholding)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Инвертируем, если номер светлый на тёмном фоне
    if np.mean(binary) > 128:
        binary = 255 - binary

    # Морфологическое закрытие для соединения компонентов
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Находим контуры
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_plate = None
    max_area = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)

        # Проверяем пропорции и размер
        if 2.5 < aspect_ratio < 6.0 and area > 1000:
            if area > max_area:
                max_area = area
                best_plate = (x, y, w, h)

    return best_plate