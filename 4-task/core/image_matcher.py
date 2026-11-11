import cv2
import numpy as np


def get_sift_features(image: np.ndarray):
    """
    Извлекает ключевые точки и дескрипторы SIFT из изображения.
    """
    # SIFT работает только с grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_two_images(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """
    Сопоставляет два изображения и возвращает количество хороших совпадений.
    Использует FLANN и тест Лоу (Lowe's ratio test).
    """
    kp1, des1 = get_sift_features(img1)
    kp2, des2 = get_sift_features(img2)

    # Если не удалось найти ключевые точки
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return 0, None

    # Настраиваем FLANN (Fast Library for Approximate Nearest Neighbors)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Находим 2 ближайших совпадения для каждого дескриптора
    matches = flann.knnMatch(des1, des2, k=2)

    # Применяем тест Лоу (Lowe's ratio test)
    # Оставляем только те совпадения, где лучшее значительно лучше второго
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return len(good_matches), good_matches


def find_most_similar_pair(images: list) -> tuple:
    """
    Находит два наиболее похожих изображения в списке.
    Сравнивает все возможные пары и выбирает ту, у которой больше всего хороших совпадений.

    Returns:
        (index1, index2, score, good_matches)
    """
    max_score = -1
    best_pair = (0, 1)
    best_matches = None

    # Перебираем все уникальные пары (i, j)
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            score, matches = match_two_images(images[i], images[j])

            if score > max_score:
                max_score = score
                best_pair = (i, j)
                best_matches = matches

    return best_pair[0], best_pair[1], max_score, best_matches