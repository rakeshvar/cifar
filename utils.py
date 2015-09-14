import pickle
import numpy as np


def load(name, dtype="float32"):
    with open(name, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    x = data["data"] / 255
    n = x.shape[0]
    x = x.reshape((n, 3, 32, 32))
    return x.astype(dtype), np.array(data["fine_labels"])


def pad_images(data_x, width, val):
    return np.pad(data_x,
                  ((0, 0), (0, 0), (width, width), (width, width)),
                  "constant", constant_values=val)


def data_info(data_x, data_y):
    print(
        "X (samples, dimensions): {} {}KB\n"
        "X (min, max) : {} {}\n"
        "Y (samples, dimensions): {} {}KB\n"
        "Y (min, max) : {} {}".format(data_x.shape, data_x.nbytes // 1000,
                                      data_x.min(), data_x.max(),
                                      data_y.shape, data_y.nbytes // 1000,
                                      data_y.min(), data_y.max()))


def load_pad_info(name, width=1, val=0):
    imgs, labels = load(name)
    imgs = pad_images(imgs, width, val)
    data_info(imgs, labels)
    return imgs, labels


labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
          'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
          'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
          'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
          'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
          'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
          'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
          'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
          'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
          'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
          'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
          'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
          'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
          'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
          'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
          'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

coarse_labels = ['aquatic_mammals', 'fish', 'flowers', 'food_containers',
                 'fruit_and_vegetables', 'household_electrical_devices',
                 'household_furniture', 'insects',
                 'large_carnivores', 'large_man-made_outdoor_things',
                 'large_natural_outdoor_scenes',
                 'large_omnivores_and_herbivores',
                 'medium_mammals', 'non-insect_invertebrates', 'people',
                 'reptiles', 'small_mammals', 'trees', 'vehicles_1',
                 'vehicles_2']

fine_to_coarse = np.array(
    [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11,
     5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19,
     8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3,
     2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19,
     2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13])