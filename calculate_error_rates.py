from collections import Counter, defaultdict
import pickle
from pprint import pprint
import sys
from theanet.neuralnet import NeuralNet
import utils
import theano as th
import numpy as np


def share(data, dtype=th.config.floatX, borrow=True):
    return th.shared(np.asarray(data, dtype), borrow=borrow)

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
trin_x, trin_y = utils.load_pad_info("train", pad_width, 0)
test_x, test_y = utils.load_pad_info("test", pad_width, 0)

############################################# Init Network
params['training_params']['BATCH_SZ'] = batch_sz = 100
ntwk = NeuralNet(**params)

############################################# Read glyphs & classify
coarse = utils.fine_to_coarse


def test_wrapper(tester, truth):
    print("Classifying...")
    fine_errors = np.zeros((100, 100), dtype="uint")
    coarse_errors = np.zeros((20, 20), dtype="uint")
    sym_err, bit_err, n = 0., 0., len(truth)//batch_sz
    for i in range(n):
        symdiff, bitdiff, feats, ypred = tester(i)
        sym_err += symdiff
        bit_err += bitdiff

        for y, yh in zip(truth[i*batch_sz:(i+1)*batch_sz], ypred):
            fine_errors[y][yh] += 1
            coarse_errors[coarse[y]][coarse[yh]] += 1

        print(i*batch_sz)

    sym_err /= n
    bit_err /= n

    return sym_err, bit_err, fine_errors, coarse_errors

def printrow(head, row, tail=""):
    print("{:3}".format(str(head)), end=' ')
    for elem in row:
        print("{:3d}".format(elem), end=' ')
    print(tail)

def printddc(ddc, labels):
    n = ddc.shape[0]
    printrow("", range(n), "<- false class labels | (down) true class labels")
    for i in range(n):
        printrow(i, ddc[i], labels[i])

    counts = np.sum(ddc, axis=1)
    printrow("tot", counts)
    printrow("err", counts-np.diag(ddc))
    printrow("%rr", (100*(counts-np.diag(ddc))/counts).astype("int"))
    print("Avg. Err. Rate: ", 1 - np.trace(ddc)/np.sum(ddc))

def test_wrapper_wrapper(fn, truths):
    s, p, f, c = test_wrapper(fn, truths)
    printddc(f, utils.labels)
    printddc(c, utils.coarse_labels)
    print("\nAvg. Fine Error Rate: {:.2%} "
          "\nProb. of True class:{:.2%}".format(s, p))

print("Test Errors")
test_fn_te = ntwk.get_test_model(share(test_x), share(test_y, "int32"),
                                 preds_feats=True)
test_wrapper_wrapper(test_fn_te, test_y)

print("Training Errors")
test_fn_tr = ntwk.get_test_model(share(trin_x), share(trin_y, "int32"),
                                 preds_feats=True)
test_wrapper_wrapper(test_fn_tr, trin_y)
