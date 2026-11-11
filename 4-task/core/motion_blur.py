import cv2
import numpy as np


def apply_motion_blur_to_moving_objects(frame: np.ndarray, prev_frame: np.ndarray,
                                         blur_strength: int) -> np.ndarray:
    """
    Размывает только движущиеся объекты на видео.
    Использует разность кадров для детекции движения.
    """
    # Вычисляем разность между текущим и предыдущим кадром
    diff = cv2.absdiff(frame, prev_frame)
    # Пороговая обработка для получения маски движения
    _, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    # Применяем размытие только к движущимся областям
    blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
    result = np.where(motion_mask[:,:,np.newaxis] > 0, blurred, frame)
    return result