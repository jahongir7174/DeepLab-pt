import csv
import multiprocessing
import os

import cv2
import numpy as np

from utils import config


def calculate_class_weight(file_name):
    _, palette = get_label_info(os.path.join(config.data_dir, 'class_dict.csv'))
    label_to_frequency = {}
    for i, label in enumerate(palette):
        label_to_frequency[i] = 0

    image = load_label(file_name)
    for i, label in enumerate(palette):
        class_mask = (image == label)
        class_mask = np.all(class_mask, axis=2)
        class_mask = class_mask.astype(np.float32)
        class_frequency = np.sum(class_mask)
        label_to_frequency[i] += class_frequency
    return label_to_frequency


def get_class_weights(file_names):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        frequencies = pool.map(calculate_class_weight, file_names)
    pool.close()
    _, palette = get_label_info(os.path.join(config.data_dir, 'class_dict.csv'))
    label_to_frequency = {}
    for i, label in enumerate(palette):
        label_to_frequency[i] = 0

    for frequency in frequencies:
        for i in range(len(palette)):
            label_to_frequency[i] += frequency[i]

    class_weights = []
    total_frequency = sum(label_to_frequency.values())
    for label, frequency in label_to_frequency.items():
        class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
        class_weights.append(class_weight)
    class_weights = np.array(class_weights, np.float32)
    return class_weights


def get_label_info(csv_path):
    palette = []
    class_names = []
    with open(csv_path, 'r') as reader:
        file_reader = csv.reader(reader, delimiter=',')
        next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            palette.append([int(row[1]), int(row[2]), int(row[3])])
    return class_names, palette


def load_image(file_name):
    path = os.path.join(config.data_dir, config.image_dir, file_name + '.png')
    return cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)


def load_label(file_name):
    path = os.path.join(config.data_dir, config.label_dir, file_name + '_L.png')
    return cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)


def save_images(images, titles, path, cols=1):
    from matplotlib import pyplot
    pyplot.axis('off')
    figure = pyplot.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = figure.add_subplot(cols, np.ceil(len(images) / float(cols)), n + 1)
        if image.ndim == 2:
            pyplot.gray()
        pyplot.imshow(image)
        pyplot.axis("off")
        a.set_title(title)
    figure.set_size_inches(np.array(figure.get_size_inches()) * len(images))
    pyplot.savefig(path)
    pyplot.close(figure)


def one_hot_it(label, palette):
    one_hot = []
    for colour in palette:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1).astype('int32')
        one_hot.append(class_map)
    one_hot = np.stack(one_hot, axis=-1).argmax(-1)  # CrossEntropyLoss
    return one_hot


def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, palette):
    x = np.array(palette)[image.astype(int)]
    return x


def random_hsv(image, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    dtype = image.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB, dst=image)


def resize(image, label, color=(0, 0, 0)):
    h, w = image.shape[:2]

    scale = min(config.image_size / w, config.image_size / h)

    image = cv2.resize(image, (int(w * scale), int(h * scale)))
    label = cv2.resize(label, (int(w * scale), int(h * scale)))

    dw = config.image_size - int(w * scale)
    dh = config.image_size - int(h * scale)

    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, label


def random_augmentation(image, label):
    if np.random.uniform(0, 1) > 0.5:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
    if np.random.uniform(0, 1) > 0.5:
        factor = 1.0 + np.random.uniform(-0.1, 0.1)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        image = cv2.LUT(image, table)
    if np.random.uniform(0, 1) > 0.5:
        angle = np.random.uniform(-20, 20)
        matrix = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
        label = cv2.warpAffine(label, matrix, (label.shape[1], label.shape[0]), flags=cv2.INTER_NEAREST)

    return image, label


def random_crop(image, label, crop_height=config.image_size, crop_width=config.image_size):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Dimension Exception')

    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = np.random.randint(0, image.shape[1] - crop_width)
        y = np.random.randint(0, image.shape[0] - crop_height)

        if len(label.shape) == 3:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width, :]
        else:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width]
    else:
        return resize(image, label)
