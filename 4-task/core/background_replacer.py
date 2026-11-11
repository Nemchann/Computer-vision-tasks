import cv2
import numpy as np


def replace_background(frame: np.ndarray, background_image: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Заменяет фон в кадре видео на статичное изображение.

    Args:
        frame: Исходный кадр видео (BGR)
        background_image: Статичное изображение для нового фона (BGR)
        fg_mask: Маска переднего плана от BackgroundSubtractor (0, 127 или 255)

    Returns:
        Кадр с заменённым фоном
    """
    # 1. Приводим маску к бинарному виду (0 и 255), убираем тени (127)
    # Всё, что > 0 (и объекты, и тени), считаем передним планом.
    # Если хочешь убрать тени, используй порог 200: _, binary_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    _, binary_mask = cv2.threshold(fg_mask, 1, 255, cv2.THRESH_BINARY)

    # 2. Подгоняем размер фона под размер кадра видео
    h, w = frame.shape[:2]
    bg_resized = cv2.resize(background_image, (w, h))

    # 3. Создаём инвертированную маску для фона (где был передний план - там будет 0)
    mask_inv = cv2.bitwise_not(binary_mask)

    # 4. Вырезаем передний план из исходного кадра
    fg = cv2.bitwise_and(frame, frame, mask=binary_mask)

    # 5. Вырезаем фон из нового изображения (по инвертированной маске)
    bg = cv2.bitwise_and(bg_resized, bg_resized, mask=mask_inv)

    # 6. Складываем передний план и новый фон
    result = cv2.add(fg, bg)

    return result