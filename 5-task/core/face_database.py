import os
import json
import time
import cv2
import numpy as np
from pathlib import Path


class FaceDatabase:
    def __init__(self, db_path: str = "data/database.json", faces_dir: str = "data/faces"):
        """
        Инициализирует базу данных лиц.
        Автоматически создаёт необходимые папки, если их нет.
        """
        self.db_path = Path(db_path)
        self.faces_dir = Path(faces_dir)

        # Создаём директорию для хранения фото, если её нет
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.profiles = []
        self.load()

    def load(self):
        """Загружает профили из JSON файла."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.profiles = json.load(f)
            except Exception as e:
                print(f"Ошибка загрузки базы данных: {e}")
                self.profiles = []
        else:
            self.profiles = []

    def save(self):
        """Сохраняет профили в JSON файл."""
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.profiles, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка сохранения базы данных: {e}")

    def add_person(self, name: str, image: np.ndarray) -> tuple:
        """
        Добавляет нового человека в базу.

        Args:
            name: Имя человека
            image: Изображение лица (numpy array, BGR)
            encoding: 128-D вектор дескриптора

        Returns:
            (success: bool, message: str)
        """
        # Проверяем, нет ли уже такого имени
        if any(p['name'].lower() == name.lower() for p in self.profiles):
            return False, f"Человек с именем '{name}' уже существует в базе."

        # Генерируем уникальное имя файла
        timestamp = int(time.time())
        safe_name = name.replace(" ", "_").replace("/", "_")
        filename = f"{safe_name}_{timestamp}.jpg"
        filepath = self.faces_dir / filename

        # Сохраняем изображение
        success = cv2.imwrite(str(filepath), image)
        if not success:
            return False, "Не удалось сохранить изображение лица."

        # Создаём запись профиля
        profile = {
            "id": timestamp,
            "name": name,
            "image_path": str(filepath)
        }

        self.profiles.append(profile)
        self.save()
        return True, "Профиль успешно добавлен."

    def get_all_profiles(self) -> list:
        """Возвращает список всех профилей (без дескрипторов, для UI)."""
        return [{"name": p["name"], "image_path": p["image_path"]} for p in self.profiles]

    def get_profile_by_name(self, name: str) -> dict:
        """Возвращает полный профиль по имени (с дескриптором)."""
        for p in self.profiles:
            if p["name"].lower() == name.lower():
                return p
        return None

    def remove_person(self, name: str) -> tuple:
        """
        Удаляет человека из базы и его фотографию.

        Returns:
            (success: bool, message: str)
        """
        for i, p in enumerate(self.profiles):
            if p["name"].lower() == name.lower():
                # Удаляем файл с диска
                try:
                    os.remove(p["image_path"])
                except FileNotFoundError:
                    pass  # Файла уже нет, это не критично

                # Удаляем из списка
                del self.profiles[i]
                self.save()
                return True, f"Профиль '{name}' удалён."

        return False, f"Профиль '{name}' не найден."