import matplotlib.pyplot as plt
import numpy as np

def build_rgb_histogram(image):
    image_array = np.array(image)

    red = image_array[:, :, 0]
    green = image_array[:, :, 1]
    blue = image_array[:, :, 2]

    red_histogram = np.bincount(red.flatten(), minlength=256)
    green_histogram = np.bincount(green.flatten(), minlength=256)
    blue_histogram = np.bincount(blue.flatten(), minlength=256)

    return red_histogram, green_histogram, blue_histogram

def show_rgb_histogram(image):
    red, green, blue = build_rgb_histogram(image)

    plt.figure(figsize=(8, 5))

    plt.plot(red, label="Red", color="red")
    plt.plot(green, label="Green", color="green")
    plt.plot(blue, label="Blue", color="blue")

    plt.title("RGB-гистограмма")
    plt.xlabel("Уровень яркости")
    plt.ylabel("Количество пикселей")

    plt.xlim(0, 255)
    plt.legend()
    plt.grid()

    plt.show()

def build_grayscale_histogram(image):
    image_array = np.array(image)

    histogram = np.bincount(
        image_array.flatten(),
        minlength=256
    )

    return histogram

def show_grayscale_histogram(image):
    histogram = build_grayscale_histogram(image)

    plt.figure(figsize=(8, 5))

    plt.plot(histogram)

    plt.title("Гистограмма изображения в градациях серого")
    plt.xlabel("Уровень яркости")
    plt.ylabel("Количество пикселей")

    plt.xlim(0, 255)
    plt.grid()

    plt.show()