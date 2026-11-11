from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from core.utils import numpy_to_qimage
import numpy as np


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