import cv2
import numpy as np


class OpticalFlowProcessor:
    """
    Класс для вычисления и визуализации плотного оптического потока (метод Фарнебека).
    """

    def __init__(self):
        self.prev_gray = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Вычисляет оптический поток между предыдущим и текущим кадром.

        Args:
            frame: Текущий кадр видео (BGR)

        Returns:
            Изображение в формате HSV->BGR, визуализирующее поток.
        """
        # 1. Конвертируем кадр в оттенки серого
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Если это первый кадр, просто сохраняем его и возвращаем оригинал
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return frame.copy()

        # 2. Вычисляем плотный оптический поток (метод Фарнебека)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, curr_gray, None,
            pyr_scale=0.5,  # Пирамидальный масштаб (0.5 - стандарт)
            levels=3,  # Количество уровней пирамиды
            winsize=15,  # Размер окна усреднения (чем больше, тем крупнее движения захватывает)
            iterations=3,  # Количество итераций на каждом уровне
            poly_n=5,  # Размер пиксельной окрестности для полиномиального разложения
            poly_sigma=1.2,  # Стандартное отклонение Гаусса для сглаживания производных
            flags=0
        )

        # 3. Визуализация потока в цветовом пространстве HSV
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255  # Насыщенность (Saturation) всегда максимальна для яркости цветов

        # Вычисляем магнитуду (скорость) и угол (направление) векторов потока
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Hue (Оттенок) = Направление движения (от 0 до 180 градусов)
        hsv[..., 0] = ang * 180 / np.pi / 2
        # Value (Яркость) = Скорость движения (нормализована от 0 до 255)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # 4. Конвертируем HSV обратно в BGR для отображения в Qt
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Сохраняем текущий кадр как предыдущий для следующего шага
        self.prev_gray = curr_gray

        return bgr_flow

    def reset(self):
        """Сбрасывает состояние (предыдущий кадр)."""
        self.prev_gray = None