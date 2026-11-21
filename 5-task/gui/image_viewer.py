from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
import cv2


def numpy_to_qimage(image: np.ndarray):
    """
    Конвертирует numpy-массив (OpenCV формат BGR) в QImage для отображения в Qt.
    """
    if image is None:
        return None

    if len(image.shape) == 2:
        # Grayscale изображение
        height, width = image.shape
        bytes_per_line = width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    elif len(image.shape) == 3:
        height, width, channels = image.shape

        if channels == 3:
            # Конвертируем BGR в RGB для Qt
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * width
            qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        elif channels == 4:
            # RGBA изображение
            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            bytes_per_line = 4 * width
            qimage = QImage(rgba_image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        else:
            raise ValueError(f"Неподдерживаемое количество каналов: {channels}")
    else:
        raise ValueError(f"Неподдерживаемая форма массива: {image.shape}")

    return qimage.copy()


class ImageViewer(QWidget):
    """Виджет для отображения изображения с заголовком."""

    def __init__(self, title: str = "Изображение", parent=None):
        super().__init__(parent)
        self.title = title
        self._image = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label_title = QLabel(self.title)
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet(
            "font-weight: bold; font-size: 14px; padding: 5px; "
            "background-color: #f0f0f0; border: 1px solid #ccc;"
        )

        self.label_image = QLabel()
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setMinimumSize(300, 300)
        self.label_image.setStyleSheet(
            "background-color: #2b2b2b; border: 1px solid #555; "
            "color: #888; font-size: 14px;"
        )
        self.label_image.setText("Нет изображения")

        layout.addWidget(self.label_title)
        layout.addWidget(self.label_image, stretch=1)

    def set_image(self, image: np.ndarray):
        """Устанавливает изображение для отображения."""
        if image is None:
            self._image = None
            self.label_image.setText("Нет изображения")
            return

        self._image = image.copy()
        self._update_display()

    def _update_display(self):
        """Перерисовывает изображение."""
        if self._image is None:
            return

        qimage = numpy_to_qimage(self._image)
        if qimage is None:
            return

        pixmap = QPixmap.fromImage(qimage)

        scaled_pixmap = pixmap.scaled(
            self.label_image.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label_image.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """При изменении размера окна — перерисовываем изображение."""
        super().resizeEvent(event)
        self._update_display()

    def get_image(self) -> np.ndarray:
        """Возвращает текущее изображение."""
        if self._image is None:
            return None
        return self._image.copy()

    def clear(self):
        """Очищает изображение."""
        self._image = None
        self.label_image.setText("Нет изображения")