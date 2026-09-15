from tkinter import filedialog, Tk, Label, Button, Scale, Frame
from image_processor import load_image, change_brightness, linear_correction, nonlinear_correction, rotate_90, get_exif_data, to_grayscale, change_contrast, get_image_info, get_color_depth, change_saturation, save_image
from histogram import show_rgb_histogram, show_grayscale_histogram
from PIL import ImageTk


original_image = None
current_image = None

# Общие функции

# Отображение изображения на экране
def display_image(image):
    display_image = image.copy()
    display_image.thumbnail((850, 850))

    tk_img = ImageTk.PhotoImage(display_image)

    image_label.config(image=tk_img)
    image_label.image = tk_img

# Открыть изображение из проводника
def open_image():
    global original_image, current_image, current_file_path

    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[
            ("Изображения", "*.jpg *.jpeg *.png *.gif *.webp *.tiff *.bmp"),
            ("Все файлы", "*.*")
        ]
    )

    if file_path:
        print("Выбран файл:", file_path)
        current_file_path = file_path

        original_image = load_image(file_path)
        current_image = original_image.copy()

        display_image(current_image)

# Сохранить изображение
def save_current_image():
    if current_image is None:
        return

    file_path = filedialog.asksaveasfilename(
        title="Сохранить изображение",
        defaultextension=".png",
        filetypes=[
            ("PNG", "*.png"),
            ("JPEG", "*.jpg"),
            ("Все файлы", "*.*")
        ]
    )

    if file_path:
        save_image(file_path, current_image)


# Отменить все действия
def reset_image():
    global current_image

    if original_image is None:
        return

    current_image = original_image.copy()
    brightness_scale.set(0)
    saturation_scale.set(0)
    contrast_scale.set(0)
    display_image(current_image)

# Информация об изображении

# Показать информацию об изображении
def show_image_info():
    if original_image is None:
        return

    info = get_image_info(
        original_image,
        current_file_path
    )

    text = ""

    for name, value in info.items():
        text += f"{name}: {value}\n"

    info_label.config(text=text)

# Показать EXIF
def show_exif_info():
    if original_image is None:
        return

    exif_info = get_exif_data(
        original_image
    )

    text = ""

    for name, value in exif_info.items():
        text += f"{name}: {value}\n"

    exif_label.config(text=text)


# Функции редактирования изображения
# Применить черно-белую градацию
def make_grayscale():
    global current_image

    current_image = to_grayscale(current_image)
    display_image(current_image)

# Изменить яркость
def apply_brightness():
    global current_image

    if original_image is None:
        return

    value = brightness_scale.get()

    current_image = change_brightness(
        original_image,
        value
    )

    display_image(current_image)

# Изменить контраст
def apply_contrast():
    global current_image

    if original_image is None:
        return

    value = contrast_scale.get()

    current_image = change_contrast(
        original_image,
        value
    )

    display_image(current_image)

# Изменить насыщенность
def apply_saturation():
    global current_image

    if original_image is None:
        return

    value = saturation_scale.get()

    current_image = change_saturation(
        original_image,
        value
    )

    display_image(current_image)

# Повернуть изображение на 90 градусов вправо
def rotate_image():
    global current_image

    if original_image is None:
        return

    current_image = rotate_90(current_image)

    display_image(current_image)

# Гистограмма

def show_histogram():
    if current_image is None:
        return

    if current_image.mode == "L":
        show_grayscale_histogram(current_image)
    else:
        show_rgb_histogram(current_image)


# Коррекции
# Линейная коррекция
def apply_linear_correction():
    global current_image

    if current_image is None:
        return

    current_image = linear_correction(current_image)
    display_image(current_image)

# Нелинейная коррекция
def apply_nonlinear_correction():
    global current_image

    if current_image is None:
        return

    current_image = nonlinear_correction(current_image)
    display_image(current_image)


# Код отображения окна Tkinter
root = Tk()
root.title("Практическая работа №1")
root.geometry("1400x900")

left_frame = Frame(root)
left_frame.pack(side="left", fill="y", padx=20, pady=20)

right_frame = Frame(root)
right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)


# Кнопки
# Загрузка изображения
btn = Button(
    left_frame,
    text="Загрузить изображение",
    width=25,
    command=open_image
)
btn.pack(pady=20)

# Сохранить изображение
btn_save = Button(
    left_frame,
    text="Сохранить изображение",
    width=25,
    command=save_current_image
)
btn_save.pack(pady=20)


# Ползунки
# Яркость
brightness_scale = Scale(
    left_frame,
    from_=-100,
    to=100,
    orient="horizontal",
    label="Яркость",
    length=250,
    command=lambda value: apply_brightness()
)
brightness_scale.set(0)
brightness_scale.pack()

# Контраст
contrast_scale = Scale(
    left_frame,
    from_=-100,
    to=100,
    orient="horizontal",
    label="Контраст",
    length=250,
    command=lambda value: apply_contrast()
)
contrast_scale.set(0)
contrast_scale.pack()

# Насыщенность
saturation_scale = Scale(
    left_frame,
    from_=-100,
    to=100,
    orient="horizontal",
    label="Насыщенность",
    length=250,
    command=lambda value: apply_saturation()
)
saturation_scale.set(0)
saturation_scale.pack()

# Еще кнопки
# Информация
info_button = Button(
    left_frame,
    text="Информация об изображении",
    width=25,
    command=show_image_info
)
info_button.pack(pady=10)

info_label = Label(
    left_frame,
    text="Информация об изображении"
)
info_label.pack(pady=10)

# EXIF
exif_button = Button(
    left_frame,
    text="EXIF",
    width=25,
    command=show_exif_info
)
exif_button.pack(pady=10)

exif_label = Label(
    left_frame,
    text="EXIF data"
)
exif_label.pack(pady=10)

# Градиент
btn_grayscale = Button(
    left_frame,
    text="Градиент",
    width=25,
    command=make_grayscale
)
btn_grayscale.pack(pady=20)

# Поворот на 90 градусов
rotate_button = Button(
    left_frame,
    text="Повернуть на 90°",
    width=25,
    command=rotate_image
)
rotate_button.pack(pady=20)

# Показ гистограммы
histogram_button = Button(
    left_frame,
    text="Показать гистограмму",
    width=25,
    command=show_histogram
)
histogram_button.pack(pady=10)

# Линейная коррекция
linear_corr_btn = Button(
    left_frame,
    text="Линейная коррекция",
    width=25,
    command=apply_linear_correction
)
linear_corr_btn.pack(pady=10)

# Нелинейная коррекция
nonlinear_corr_btn = Button(
    left_frame,
    text="Нелинейная коррекция",
    width=25,
    command=apply_nonlinear_correction
)
nonlinear_corr_btn.pack(pady=10)

# Отмена всех действий
reset_button = Button(
    left_frame,
    text="Отменить все действия",
    width=25,
    command=reset_image
)
reset_button.pack(pady=10)

# Место для самого изображения
image_label = Label(right_frame)
image_label.pack(pady=10)


root.mainloop()

