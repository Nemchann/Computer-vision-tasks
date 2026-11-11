import cv2
import numpy as np


class BackgroundSubtractor:
    """
    Класс для вычитания фона из видео.
    Использует алгоритм MOG2 (Gaussian Mixture Model).
    """

    def __init__(self, history: int = 500, var_threshold: float = 16.0,
                 detect_shadows: bool = True):
        """
        Инициализирует вычитатель фона.

        Args:
            history: Количество последних кадров для моделирования фона (по умолчанию 500)
            var_threshold: Порог дисперсии для определения "новизны" пикселя (по умолчанию 16.0)
            detect_shadows: Определять ли тени (по умолчанию True)
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )

    def apply(self, frame: np.ndarray) -> tuple:
        """
        Применяет вычитание фона к кадру.

        Args:
            frame: Текущий кадр видео (BGR)

        Returns:
            (foreground_mask, foreground_image)
            - foreground_mask: Бинарная маска переднего плана (255 = объект, 0 = фон, 127 = тень)
            - foreground_image: Изображение только с передним планом (фон чёрный)
        """
        # Применяем вычитатель фона
        fg_mask = self.bg_subtractor.apply(frame)

        # Создаём изображение только с передним планом
        # Используем битовое И для маскирования
        foreground = cv2.bitwise_and(frame, frame, mask=fg_mask)

        return fg_mask, foreground

    def reset(self):
        """Сбрасывает модель фона (полезно при смене видео)."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()