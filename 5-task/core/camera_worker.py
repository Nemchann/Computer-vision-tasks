import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from core.face_detector import detect_faces, draw_face_boxes
from core.face_matcher import FaceMatcher


class CameraWorker(QThread):
    """
    Поток для работы с веб-камерой и распознавания лиц в реальном времени.
    """

    # Сигналы для передачи данных в GUI
    frame_ready = pyqtSignal(QImage)  # Новый кадр для отображения
    status_updated = pyqtSignal(str)  # Обновление статуса
    fps_updated = pyqtSignal(float)  # Обновление FPS

    def __init__(self, face_db, camera_index: int = 0,
                 detection_scale: float = 0.5, confidence_threshold: float = 120.0):
        super().__init__()
        self.face_db = face_db
        self.camera_index = camera_index
        self.detection_scale = detection_scale  # Уменьшение для детекции
        self.confidence_threshold = confidence_threshold

        self.running = False
        self.cap = None
        self.matcher = None

        # Стабилизация: запоминаем последнее предсказание
        self.last_name = None
        self.last_confidence = None
        self.stable_frames = 0

    def run(self):
        """Основной цикл работы потока."""
        self.running = True

        # 1. Открываем камеру
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.status_updated.emit("❌ Не удалось открыть камеру")
            return

        # Устанавливаем разрешение 640x480 (быстрее, чем 1280x720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 2. Обучаем распознаватель ОДИН РАЗ
        profiles = self.face_db.get_all_profiles()
        if len(profiles) < 2:
            self.status_updated.emit("❌ В базе меньше 2 профилей — обучение невозможно")
            self.cap.release()
            return

        self.matcher = FaceMatcher()
        self.matcher.set_threshold(self.confidence_threshold)

        if not self.matcher.train(profiles):
            self.status_updated.emit("❌ Не удалось обучить распознаватель")
            self.cap.release()
            return

        self.status_updated.emit(f"✅ Распознаватель обучен на {len(profiles)} фото")

        # 3. Цикл обработки кадров
        frame_count = 0
        import time
        fps_timer = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_updated.emit("⚠️ Не удалось прочитать кадр")
                break

            frame_count += 1

            # Считаем FPS каждые 30 кадров
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_timer
                fps = 30 / elapsed if elapsed > 0 else 0
                self.fps_updated.emit(fps)
                fps_timer = time.time()

            # 4. Детекция лиц на уменьшенном изображении
            small_frame = cv2.resize(
                frame, None,
                fx=self.detection_scale,
                fy=self.detection_scale
            )
            small_faces = detect_faces(small_frame)

            # Возвращаем координаты к оригинальному размеру
            scale_inv = 1.0 / self.detection_scale
            face_locations = [
                (int(t * scale_inv), int(r * scale_inv),
                 int(b * scale_inv), int(l * scale_inv))
                for (t, r, b, l) in small_faces
            ]

            # 5. Рисуем рамки и распознаём
            result_frame = frame.copy()

            for (top, right, bottom, left) in face_locations:
                # Вырезаем лицо для распознавания
                face_crop = frame[top:bottom, left:right]
                if face_crop.size == 0:
                    continue

                # Распознаём
                name, confidence = self.matcher.predict(face_crop)

                # Стабилизация: если уверенность выше порога — показываем "Unknown"
                if confidence > self.confidence_threshold:
                    display_name = "Unknown"
                    color = (0, 0, 255)  # Красный
                else:
                    display_name = name
                    color = (0, 255, 0)  # Зелёный

                # Рисуем рамку
                cv2.rectangle(result_frame, (left, top), (right, bottom), color, 2)

                # Рисуем фон для текста (чтобы имя было читаемым)
                label_text = f"{display_name} ({confidence:.0f})"
                (text_w, text_h), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    result_frame,
                    (left, top - text_h - 10),
                    (left + text_w + 10, top),
                    color, -1
                )
                cv2.putText(
                    result_frame, label_text,
                    (left + 5, top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2
                )

            # 6. Отправляем кадр в GUI
            qimage = self._numpy_to_qimage(result_frame)
            self.frame_ready.emit(qimage)

        # Освобождение ресурсов
        self.cap.release()
        self.status_updated.emit("⏹ Камера остановлена")

    def _numpy_to_qimage(self, image: np.ndarray) -> QImage:
        """Конвертирует BGR numpy-массив в QImage для Qt."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

    def stop(self):
        """Останавливает поток."""
        self.running = False
        self.wait(2000)  # Ждём завершения до 2 секунд

    def set_confidence_threshold(self, threshold: float):
        """Обновляет порог уверенности во время работы."""
        self.confidence_threshold = threshold
        if self.matcher is not None:
            self.matcher.set_threshold(threshold)