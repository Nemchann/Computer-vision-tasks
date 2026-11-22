import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


class FaceMatcher:
    """
    Распознаватель лиц на основе LBPH с улучшенной логикой.
    """

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8,
            threshold=80.0  # Порог по умолчанию
        )
        self.is_trained = False
        self.label_to_name = {}

    def train(self, profiles: list) -> bool:
        """
        Обучает распознаватель, группируя фото по имени.
        """
        if len(profiles) < 2:
            self.is_trained = False
            return False

        # Группируем фото по имени
        name_to_images = defaultdict(list)
        for profile in profiles:
            if not Path(profile['image_path']).exists():
                continue

            image = cv2.imread(profile['image_path'])
            if image is None:
                continue

            # Переводим в grayscale и применяем CLAHE для улучшения контраста
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Ресайз до стандартного размера
            gray = cv2.resize(gray, (100, 100))

            name_to_images[profile['name']].append(gray)

        if len(name_to_images) < 2:
            self.is_trained = False
            return False

        # Создаём списки изображений и меток
        images = []
        labels = []
        self.label_to_name = {}

        label = 0
        for name, imgs in name_to_images.items():
            for img in imgs:
                images.append(img)
                labels.append(label)
            self.label_to_name[label] = name
            label += 1

        if len(images) < 2:
            self.is_trained = False
            return False

        # Обучаем
        self.recognizer.train(images, np.array(labels))
        self.is_trained = True
        return True

    def predict(self, image: np.ndarray) -> tuple:
        """
        Распознаёт лицо на изображении.
        """
        if not self.is_trained:
            return "Не обучено", -1.0

        # Переводим в grayscale и применяем CLAHE
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.resize(gray, (100, 100))

        # Предсказываем
        label, confidence = self.recognizer.predict(gray)
        name = self.label_to_name.get(label, "Неизвестно")

        return name, confidence

    def get_threshold(self) -> float:
        return self.recognizer.getThreshold()

    def set_threshold(self, threshold: float):
        self.recognizer.setThreshold(threshold)