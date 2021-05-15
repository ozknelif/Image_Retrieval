from myimage import MyImage
from PIL import Image
import numpy as np
import random
import pathlib

Image.LOAD_TRUNCATED_IMAGES = True

RGB_FOLDER = 'rgb'
HSV_FOLDER = 'hsv'


def read(filename, count_test, count_train):
    images = []
    for path in pathlib.Path('images\\' + filename).iterdir():
        if path.is_file():
            image = (Image.open(path))
            images.append(MyImage(image, path.name))

    train = []
    test = []
    iter_array = []
    count = 0
    while count < count_train:
        iter = random.randint(0, len(images) - 1)
        if (iter not in iter_array):
            train.append(images[iter])
            iter_array.append(iter)
            count += 1

    count = 0
    while count < count_test:
        iter = random.randint(0, len(images) - 1)
        if (iter not in iter_array):
            test.append(images[iter])
            iter_array.append(iter)
            count += 1

    return train, test


def hsv_histogram(images):
    for img in images:
        img_hsv = img.image.convert('RGB')
        i = 0
        while i < img.width:
            j = 0
            while j < img.height:
                r, g, b = img_hsv.getpixel((i, j))
                h = rgb2hue(r, g, b)
                img.histogram_h[int(h)] = img.histogram_h[int(h)] + (1.0 / (img.width * img.height))
                j += 1
            i += 1


def rgb_histogram(images):
    for img in images:
        img_rgb = img.image.convert('RGB')
        i = 0
        while i < img.width:
            j = 0
            while j < img.height:
                r, g, b = img_rgb.getpixel((i, j))
                img.histogram_r[r] = img.histogram_r[r] + (1.0 / (img.width * img.height))
                img.histogram_g[g] = img.histogram_g[g] + (1.0 / (img.width * img.height))
                img.histogram_b[b] = img.histogram_b[b] + (1.0 / (img.width * img.height))
                j += 1
            i += 1


def rgb_similarity(train, test):
    dist_list = np.zeros((len(train), 2), dtype=object)
    for i, train in enumerate(train):
        r_dist = np.sqrt(np.sum([np.power(a_i - b_i, 2) for a_i, b_i in zip(test.histogram_r, train.histogram_r)]))
        g_dist = np.sqrt(np.sum([np.power(a_i - b_i, 2) for a_i, b_i in zip(test.histogram_g, train.histogram_g)]))
        b_dist = np.sqrt(np.sum([np.power(a_i - b_i, 2) for a_i, b_i in zip(test.histogram_b, train.histogram_b)]))
        dist = np.sqrt(r_dist ** 2 + g_dist ** 2 + b_dist ** 2)
        dist_list[i][0] = np.float64(dist)
        dist_list[i][1] = train
    dist_list = dist_list[dist_list[:, 0].argsort()]
    test.image.save(RGB_FOLDER + '\\' + test.name)
    for img in dist_list[0: 5, ]:
        img[1].image.save(RGB_FOLDER + '\\' + test.name)


def hsv_similarity(train, test):
    dist_list = np.zeros((len(train), 2), dtype=object)
    for i, train in enumerate(train):
        h_dist = np.sqrt(np.sum([np.power(a_i - b_i, 2) for a_i, b_i in zip(test.histogram_h, train.histogram_h)]))
        dist_list[i][0] = np.float64(h_dist)
        dist_list[i][1] = train
    dist_list = dist_list[dist_list[:, 0].argsort()]
    test.image.save(HSV_FOLDER + '\\' + test.name)
    for img in dist_list[0: 5, ]:
        img[1].image.save(HSV_FOLDER + '\\' + test.name)


def rgb2hue(r, g, b):
    r = r / 255
    g = g / 255
    b = b / 255
    maximum = max(r, g, b)
    minimum = min(r, g, b)
    if minimum == maximum:
        return 0
    if maximum == r:
        h = (g - b) / (maximum - minimum)
    elif maximum == g:
        h = 2.0 + (b - r) / (maximum - minimum)
    else:
        h = 4.0 + (r - g) / (maximum - minimum)

    h *= 60
    h = np.round(h)
    if h > 0:
        return h
    else:
        return h + 360.0


def save(file_path, test, dist_list):
    test.image.save(file_path + '\\' + test.name)
    for img in dist_list:
        img[1].image.save(file_path + '\\' + test.name)


train_camel, test_camel = read('028.camel', 5, 25)
train_dog, test_dog = read('056.dog', 5, 25)
train_dolphin, test_dolphin = read('057.dolphin', 5, 25)
train_giraffe, test_giraffe = read('084.giraffe', 5, 25)
train_goose, test_goose = read('089.goose', 5, 25)
train_horse, test_horse = read('105.horse', 5, 25)

rgb_histogram(train_camel)
rgb_histogram(train_dog)
rgb_histogram(train_dolphin)
rgb_histogram(train_giraffe)
rgb_histogram(train_goose)
rgb_histogram(train_horse)

rgb_histogram(test_camel)
rgb_histogram(test_dog)
rgb_histogram(test_dolphin)
rgb_histogram(test_giraffe)
rgb_histogram(test_goose)
rgb_histogram(test_horse)

train_images = train_camel + train_dog + train_dolphin + train_giraffe + train_goose + train_horse

for test in test_camel:
    rgb_similarity(train_images, test)

for test in test_dolphin:
    rgb_similarity(train_images, test)

for test in test_dog:
    rgb_similarity(train_images, test)

for test in test_giraffe:
    rgb_similarity(train_images, test)

for test in test_goose:
    rgb_similarity(train_images, test)

for test in test_horse:
    rgb_similarity(train_images, test)

hsv_histogram(train_camel)
hsv_histogram(train_dog)
hsv_histogram(train_dolphin)
hsv_histogram(train_giraffe)
hsv_histogram(train_goose)
hsv_histogram(train_horse)

hsv_histogram(test_camel)
hsv_histogram(test_dog)
hsv_histogram(test_dolphin)
hsv_histogram(test_giraffe)
hsv_histogram(test_goose)
hsv_histogram(test_horse)

train_images = train_camel + train_dog + train_dolphin + train_giraffe + train_goose + train_horse

for test in test_camel:
    hsv_similarity(train_images, test)

for test in test_dolphin:
    hsv_similarity(train_images, test)

for test in test_dog:
    hsv_similarity(train_images, test)

for test in test_giraffe:
    hsv_similarity(train_images, test)

for test in test_goose:
    hsv_similarity(train_images, test)

for test in test_horse:
    hsv_similarity(train_images, test)
