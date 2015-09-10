import os
import pickle
import sys
from PIL import Image as im
from data.categorize import tile_raster_images
import numpy as np
from theanet.neuralnet import NeuralNet
import utils

############################################# Arguments

if len(sys.argv) < 2:
    print("Usage:\n\t{0} neuralnet_params.pkl".format(sys.argv[0]))
    sys.exit()

print("\nLoading the network info ...")
nnet_prms_file_name = sys.argv[1]
with open(nnet_prms_file_name, 'rb') as nnet_prms_file:
    params = pickle.load(nnet_prms_file)

layers = params['layers']
tr_prms = params['training_params']
allwts = params['allwts']

############################################# Load Data
print("\nLoading the data ...")
try:
    img_sz = layers[0][1]["img_sz"]
except KeyError:
    img_sz = layers[0][1]["img_sz"] = 32

pad_width = (img_sz-32)//2
#trin_x, trin_y = utils.load_pad_info("train", pad_width, 0)
test_x, test_y = utils.load_pad_info("test", pad_width, 0)
test_col = (np.rollaxis(test_x, 1, 4)*255).astype("uint8")


############################################# Init Network
params['training_params']['BATCH_SZ'] = 1
ntwk = NeuralNet(**params)
tester = ntwk.get_data_test_model(go_nuts=True)

############################################# Image saver
dir_name = os.path.basename(nnet_prms_file_name)[:-4] + '/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

namer = (dir_name + '{}_{:02d}.png').format
    # Usage:namer(info, i)

def saver(outs, info, debug=True):
    for i, out in enumerate(outs):
        global_normalize = False
        if out.ndim == 2:
            n_nodes = out.shape[1]
            w = n_nodes // int(np.sqrt(n_nodes))
            h = np.ceil(float(n_nodes) / w)
            extra = np.full((1, w*h-n_nodes), 0)
            out = np.concatenate((out, extra), 1).reshape((1, h, w))
        elif out.ndim == 4:
            out = out[0]
            if out.shape[-1] * out.shape[-2] < 65:
                global_normalize = True

        if debug:
            print("{:6.2f} {:6.2f} {:6.2f} {} GNM:{}".format(
                out.max(), out.mean(), out.min(), out.shape,
                global_normalize))

        im.fromarray(tile_raster_images(out, zm=2,
                                        make_white=True,
                                        global_normalize=True)
        ).save(namer(info, i), compress_level=1)

    if debug:
        print()

############################################# Read glyphs & classify
print("Classifying...")
for i in range(100):
    img = test_x[i:i + 1, ...]
    logprobs_or_feats, preds, *layer_outs = tester(img)

    y, y1 = test_y[i], preds[0]
    true_p, true_label = np.exp(logprobs_or_feats[0][y]), utils.labels[y]
    info = "test_{:03d}_{}_{:02.0f}".format(i, true_label, true_p*100)
    if y != y1:
        pred_p, pred_label = np.exp(logprobs_or_feats[0][y1]), utils.labels[y1]
        info += "_as_{}_{:02.0f}".format(pred_label, pred_p*100)
    print("Image: ", info,)
    saver(layer_outs, info, False)
    im.fromarray(test_col[i]).save(namer(info, 0)) # Overwrites input layer

print("Output images in :", dir_name)
