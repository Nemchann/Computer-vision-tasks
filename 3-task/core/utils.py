import cv2
import numpy as np
from PyQt5.QtGui import QImage

def numpy_to_qimage(image: np.ndarray) -> QImage:
    """
    Конвертирует numpy-массив (OpenCV формат) в QImage для отображения в Qt.
    Поддерживает grayscale и цветные (BGR) изображения.
    """
    if image is None:
        return QImage()

    # Определяем формат по количеству каналов
    if len(image.shape) == 2:
        # Grayscale изображение
        height, width = image.shape
        bytes_per_line = width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    elif len(image.shape) == 3:
        height, width, channels = image.shape
        if channels == 3:
            # BGR → RGB (OpenCV использует BGR, а Qt — RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * width
            qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        elif channels == 4:
            # BGRA → RGBA
            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            bytes_per_line = 4 * width
            qimage = QImage(rgba_image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        else:
            raise ValueError(f"Неподдерживаемое количество каналов: {channels}")
    else:
        raise ValueError(f"Неподдерживаемая форма массива: {image.shape}")

    # ВАЖНО: нужно сделать копию данных, иначе QImage будет ссылаться
    # на временный буфер numpy, который может быть освобождён
    return qimage.copy()

def imread_unicode(filepath: str) -> np.ndarray:
    """
    Загружает изображение, поддерживая Unicode пути (кириллица и т.д.) на Windows.
    Использует cv2.imdecode() + np.fromfile() вместо cv2.imread().
    """
    # Читаем файл в numpy-массив байтов
    data = np.fromfile(filepath, dtype=np.uint8)

    # Декодируем изображение из байтов
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Не удалось декодировать изображение: {filepath}")

    return image


def imwrite_unicode(filepath: str, image: np.ndarray) -> bool:
    """
    Сохраняет изображение, поддерживая Unicode пути (кириллица и т.д.) на Windows.
    Использует cv2.imencode() + np.tofile() вместо cv2.imwrite().
    """
    # Определяем формат по расширению файла
    ext = filepath.lower().split('.')[-1]

    # Маппинг расширений на форматы OpenCV
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

    # Кодируем изображение в буфер
    success, buffer = cv2.imencode(format_str, image)

    if not success:
        return False

    # Записываем буфер в файл
    buffer.tofile(filepath)
    return True


def load_image_as_grayscale(filepath: str) -> np.ndarray:
    """Загружает изображение и сразу конвертирует в grayscale."""
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {filepath}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale


def load_image_original(filepath: str) -> np.ndarray:
    """Загружает изображение в оригинальном виде (BGR или BGRA)."""
    image = imread_unicode(filepath)
    return image