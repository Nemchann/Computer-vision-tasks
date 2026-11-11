# import cv2
# import numpy as np
#
#
# def compute_optical_flow(prev_frame: np.ndarray, current_frame: np.ndarray) -> np.ndarray:
#     """
#     Вычисляет и визуализирует оптический поток (метод Farneback).
#     Возвращает изображение со стрелками/цветами, показывающими движение.
#     """
#     flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
#                                          0.5, 3, 15, 3, 5, 1.2, 0)
#     # Визуализация через hsv
#     hsv = np.zeros_like(current_frame)
#     hsv[..., 1] = 255
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     hsv[..., 0] = ang * 180 / np.pi / 2
#     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#     result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     return result