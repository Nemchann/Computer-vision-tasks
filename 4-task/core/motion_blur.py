import cv2
import numpy as np


class MotionBlurFilter:
    """
    Фильтр для размытия движущихся объектов на видео.
    Использует разность кадров для детекции движения.
    """

    def __init__(self, blur_strength: int = 15):
        self.blur_strength = blur_strength
        self.prev_frame = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Применяет размытие к движущимся объектам в кадре."""
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return frame.copy()

        # Детекция движения
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, curr_gray)

        _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Морфологические операции
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

        # Размытие
        blurred_frame = cv2.GaussianBlur(frame, (self.blur_strength, self.blur_strength), 0)

        # Композитинг
        mask_inv = cv2.bitwise_not(motion_mask)
        motion_blurred = cv2.bitwise_and(blurred_frame, blurred_frame, mask=motion_mask)
        static_clear = cv2.bitwise_and(frame, frame, mask=mask_inv)
        result = cv2.add(motion_blurred, static_clear)

        self.prev_frame = frame.copy()
        return result

    def reset(self):
        self.prev_frame = None

    def set_blur_strength(self, strength: int):
        if strength < 3:
            strength = 3
        if strength % 2 == 0:
            strength += 1
        self.blur_strength = strength

    def process_and_save_video(self, input_path: str, output_path: str,
                               progress_callback=None) -> bool:
        """
        Обрабатывает видео и сохраняет результат в файл.

        Args:
            input_path: Путь к исходному видео
            output_path: Путь для сохранения результата
            progress_callback: Функция для отображения прогресса (current, total)

        Returns:
            True если успешно, False иначе
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False

        # Получаем параметры видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Создаём VideoWriter
        # Используем codec MJPG для совместимости, или XVID
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # или 'XVID', 'MJPG'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            return False

        self.reset()
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Применяем размытие
                result = self.apply(frame)

                # Записываем кадр
                out.write(result)

                frame_count += 1

                # Обновляем прогресс
                if progress_callback and frame_count % 10 == 0:
                    progress_callback(frame_count, total_frames)

            return True
        finally:
            cap.release()
            out.release()