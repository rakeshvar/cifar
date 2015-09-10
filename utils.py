import pickle
import numpy as np

def load(name, dtype="float32"):
    with open(name, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    x = data["data"]/255
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
    return  imgs, labels

labels= ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
      'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
      'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
      'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
      'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
      'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
      'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
      'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
      'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
      'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit',
      'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
      'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
      'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
      'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
      'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']