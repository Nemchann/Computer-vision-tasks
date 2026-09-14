import os
import numpy as np

from PIL import Image, ImageEnhance
from PIL.ExifTags import TAGS

def load_image(file_path):
    print("Изображение загружено:", file_path)
    return Image.open(file_path)

def save_image(file_path, image):
    if file_path.lower().endswith((".jpg", ".jpeg")) and image.mode == "RGBA":
        image = image.convert("RGB")

    image.save(file_path)
    print("Сохранено")


def to_grayscale(image):
    return image.convert("L")

def change_brightness(image, slider_value):
    factor = 1.0 + (slider_value / 100.0)

    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def change_contrast(image, slider_value):
    factor = 1.0 + (slider_value / 100.0)

    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def change_saturation(image, slider_value):
    factor = 1.0 + (slider_value / 100.0)

    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def get_color_depth(image):
    if image.mode == "1":
        return 1
    elif image.mode == "L":
        return 8
    elif image.mode == "RGB":
        return 24
    elif image.mode == "RGBA":
        return 32
    elif image.mode == "P":
        return 8
    else:
        return "Неизвестно"

def get_image_info(image, file_path):
    file_size = os.path.getsize(file_path)

    return {
        "Размер на диске": f"{file_size / 1024:.2f} КБ",
        "Разрешение": f"{image.width} × {image.height}",
        "Глубина цвета": f"{get_color_depth(image)} бит",
        "Формат файла": image.format,
        "Цветовая модель": image.mode
    }

def get_exif_data(image):
    exif = image.getexif()

    if not exif:
        return {}

    exif_data = {}

    for tag_id, value in exif.items():
        tag_name = TAGS.get(tag_id, tag_id)
        exif_data[tag_name] = value

    return exif_data

def rotate_90(image):
    return image.rotate(-90, expand=True)


def linear_correction(image):
    image_array = np.array(image, dtype=np.float32)

    min_value = image_array.min()
    max_value = image_array.max()

    if min_value == max_value:
        return image.copy()

    corrected = (image_array - min_value) * 255 / (max_value - min_value)

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    return Image.fromarray(corrected)


def nonlinear_correction(image, gamma=0.5):
    image_array = np.array(image, dtype=np.float32)

    normalized = image_array / 255.0

    corrected = 255 * (normalized ** gamma)

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    return Image.fromarray(corrected)