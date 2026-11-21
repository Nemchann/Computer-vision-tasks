import cv2
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt


def numpy_to_qimage(image: np.ndarray) -> QImage:
    """
    Конвертирует numpy-массив (OpenCV формат) в QImage для отображения в Qt.
    """
    if image is None:
        return QImage()

    if len(image.shape) == 2:
        height, width = image.shape
        bytes_per_line = width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    elif len(image.shape) == 3:
        height, width, channels = image.shape
        if channels == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * width
            qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        elif channels == 4:
            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            bytes_per_line = 4 * width
            qimage = QImage(rgba_image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        else:
            raise ValueError(f"Неподдерживаемое количество каналов: {channels}")
    else:
        raise ValueError(f"Неподдерживаемая форма массива: {image.shape}")

    return qimage.copy()


def imread_unicode(filepath: str) -> np.ndarray:
    """Загружает изображение с поддержкой Unicode путей."""
    data = np.fromfile(filepath, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Не удалось декодировать изображение: {filepath}")
    return image


def imwrite_unicode(filepath: str, image: np.ndarray) -> bool:
    """Сохраняет изображение с поддержкой Unicode путей."""
    ext = filepath.lower().split('.')[-1]
    ext_to_format = {
        'jpg': '.jpg',
        'jpeg': '.jpg',
        'png': '.png',
        'bmp': '.bmp',
        'tif': '.tif',
        'tiff': '.tif',
        'webp': '.webp'
    }

    if ext not in ext_to_format:
        raise ValueError(f"Неподдерживаемый формат файла: .{ext}")

    format_str = ext_to_format[ext]
    success, buffer = cv2.imencode(format_str, image)

    if not success:
        return False

    buffer.tofile(filepath)
    return True


def load_image(filepath: str) -> np.ndarray:
    """Загружает изображение."""
    return imread_unicode(filepath)


def load_video(filepath: str) -> cv2.VideoCapture:
    """Открывает видеофайл."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {filepath}")
    return cap