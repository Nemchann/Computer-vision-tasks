import numpy as np
from sklearn.cluster import KMeans


def segment_kmeans(image: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Сегментирует цветное изображение с помощью алгоритма K-means.

    Args:
        image: Входное цветное изображение (BGR, 3 канала)
        n_clusters: Количество кластеров (от 2 до 20)

    Returns:
        Сегментированное изображение (каждый пиксель заменён центром его кластера)
    """
    # 1. Сохраняем оригинальную форму изображения, чтобы потом восстановить
    original_shape = image.shape

    # 2. Преобразуем изображение в 2D массив пикселей (строка = один пиксель [B, G, R])
    #    Например, изображение 100x100 превратится в массив из 10000 строк по 3 значения
    pixels = image.reshape(-1, 3)

    # 3. Создаём и обучаем модель K-means
    #    n_init='auto' убирает предупреждения в новых версиях scikit-learn
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(pixels)

    # 4. Заменяем каждый пиксель на координаты центра его кластера
    #    kmeans.labels_ содержит индекс кластера для каждого пикселя
    #    kmeans.cluster_centers_ содержит цвета (B, G, R) центров кластеров
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    # 5. Возвращаем массиву форму оригинального изображения
    segmented_image = segmented_pixels.reshape(original_shape)

    # 6. Округляем и приводим к типу uint8 (OpenCV требует целые числа от 0 до 255)
    return segmented_image.astype(np.uint8)