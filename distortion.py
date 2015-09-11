import ast
import os
import sys
from PIL import Image as im

import numpy as np
import theano.tensor as tt
from theano import shared, config, function
from theanet.layer import InputLayer, ElasticLayer, ColorLayer
import utils

############################################## Arguments
batch_sz = 7
n_batches = 20
n_distortions = 5
data_file = "train"
dir_name = 'distorted_images/'

if len(sys.argv) < 2:
    print("Usage:\n"
          "{} nnet_params.prms [data={}] [begin_at_index]"
          "\n\tOpens the nnet_params.prms file and the data of images and "
          "distorts {} batches (of size {}) of images {} times and writes "
          "one output image per distortion to the {} folder."
          "".format(sys.argv[0], data_file, n_batches, batch_sz,
                    n_distortions, dir_name))
    sys.exit(-1)

prms_file_name = sys.argv[1]
try:
    data_file = sys.argv[2]
except IndexError:
    pass

try:
    begin = int(sys.argv[3])
except IndexError:
    begin = None

with open(prms_file_name, 'r') as p_fp:
    params = ast.literal_eval(p_fp.read())
    layers = params['layers']
    tr_prms = params['training_params']

############################################## Load
print('Loading Data')

try:
    img_sz = layers[0][1]["img_sz"]
except KeyError:
    img_sz = layers[0][1]["img_sz"] = 32

pad_width = (img_sz-32)//2
data_x, data_y = utils.load_pad_info(data_file, pad_width, 0)
corpus_sz = data_x.shape[0]

# Print top three layers
for lyr, prms in layers[:3]:
    print(lyr)
    for param, value in prms.items():
        print("    {:20}:{}".format(param, value))
print()

############################################## Init Layer
imgs = shared(np.asarray(data_x, config.floatX), borrow=True)
x_sym = tt.tensor4('x')
net_layers, i = [], 0
lyr_look_up = {"InputLayer":InputLayer,
               "ElasticLayer":ElasticLayer,
               "ColorLayer":ColorLayer}
curr_output = x_sym

while layers[i][0] in lyr_look_up:
    layer = lyr_look_up[layers[i][0]]
    args = layers[i][1]
    args["num_maps"] = 3
    args["img_sz"] = img_sz
    net_layers.append(layer(curr_output, **args))
    i += 1
    curr_output = net_layers[-1].output
    print(str(net_layers[-1]).replace(" ", "\n    "))

deform_fn = function([x_sym], curr_output)

############################################## Naming
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def get_out_name(idx):
    name = dir_name + data_file + str(idx)
    for i in range(idx, idx+batch_sz):
        name += "-" + utils.labels[data_y[i]]
    return name + '.png'

############################################## Distort
if begin is None:
    begin = np.random.randint(corpus_sz-batch_sz*n_batches, size=1)[0]

margin = 1
img_szm = img_sz + margin
out_img = np.zeros((img_szm*(n_distortions+1)+1, img_szm*batch_sz+1, 3)) + .5


def assign_row(composite_image, image_stack, row):
    """
    Helper function to assign a stack of color images to a row of the
    composite image.
    :param composite_image: composite image
    :param image_stack: the stack of color images
    :param row: the row where it goes
    """
    image_stack = np.rollaxis(image_stack, 1, 4)
    composite_row = row*img_szm + 1
    for i in range(batch_sz):
        col = i*img_szm + 1
        composite_image[composite_row:composite_row+img_sz,
                        col:col+img_sz] = image_stack[i]


for index in range(begin, begin+n_batches*batch_sz, batch_sz):
    in_img = data_x[index : index+batch_sz]
    assign_row(out_img, in_img, 0)
    for dist in range(n_distortions):
        df_imgs = deform_fn(in_img)
        assign_row(out_img, df_imgs, dist+1)

    out_fname = get_out_name(index)
    scaled = (255*out_img).astype('uint8')
    im.fromarray(scaled).save(out_fname)

    print("Saved ", out_fname)
