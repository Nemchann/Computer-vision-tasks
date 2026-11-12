import cv2
import numpy as np


class MotionBlurFilter:
    """
    Фильтр для размытия движущихся объектов на видео.
    Использует разность кадров для детекции движения.
    """

    def __init__(self, blur_strength: int = 15):
        """
        Инициализирует фильтр размытия.

        Args:
            blur_strength: Сила размытия (размер ядра Гаусса).
                          Должно быть нечётным числом (5, 9, 15, 21, ...)
        """
        self.blur_strength = blur_strength
        self.prev_frame = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Применяет размытие к движущимся объектам в кадре.

        Args:
            frame: Текущий кадр видео (BGR)

        Returns:
            Кадр с размытыми движущимися объектами
        """
        # Если это первый кадр, просто сохраняем его и возвращаем
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return frame.copy()

        # 1. Конвертируем оба кадра в grayscale для сравнения
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)

        # 2. Вычисляем разность между кадрами
        frame_diff = cv2.absdiff(prev_gray, curr_gray)

        # 3. Применяем пороговую обработку для получения бинарной маски движения
        # Пиксели, изменившиеся больше чем на 30 единиц, считаем движением
        _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # 4. Морфологические операции для очистки маски
        # Удаляем шум и соединяем разрозненные области
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

        # 5. Применяем размытие Гаусса ко всему кадру
        blurred_frame = cv2.GaussianBlur(frame, (self.blur_strength, self.blur_strength), 0)

        # 6. Создаём финальное изображение:
        # - Где есть движение (mask > 0) — берём размытый кадр
        # - Где нет движения (mask = 0) — берём оригинальный кадр
        # Используем cv2.bitwise_and и cv2.add для композитинга

        # Инвертируем маску для фона (статичных областей)
        mask_inv = cv2.bitwise_not(motion_mask)

        # Вырезаем размытые движущиеся объекты
        motion_blurred = cv2.bitwise_and(blurred_frame, blurred_frame, mask=motion_mask)

        # Вырезаем чёткие статичные области
        static_clear = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Складываем вместе
        result = cv2.add(motion_blurred, static_clear)

        # Сохраняем текущий кадр как предыдущий для следующего вызова
        self.prev_frame = frame.copy()

        return result

    def reset(self):
        """Сбрасывает предыдущий кадр (полезно при смене видео)."""
        self.prev_frame = None

    def set_blur_strength(self, strength: int):
        """
        Устанавливает силу размытия.

        Args:
            strength: Нечётное число >= 3
        """
        if strength < 3:
            strength = 3
        if strength % 2 == 0:
            strength += 1
        self.blur_strength = strength