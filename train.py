#! /usr/bin/python
# -*- coding: utf-8 -*-
import ast
import pickle
import os
import socket
import sys
from datetime import datetime

import numpy as np
import theano as th
import theanet.neuralnet as nn

import utils
################################ HELPER FUNCTIONS ############################

def share(data, dtype=th.config.floatX, borrow=True):
    return th.shared(np.asarray(data, dtype), borrow=borrow)

class WrapOut:
    def __init__(self, use_file, name=''):
        self.name = name
        self.use_file = use_file
        if use_file:
            self.stream = open(name, 'w', 1)
        else:
            self.stream = sys.stdout

    def write(self, data):
        self.stream.write(data)

    def forceflush(self):
        if self.use_file:
            self.stream.close()
            self.stream = open(self.name, 'a', 1)

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

################################### MAIN CODE ################################

if len(sys.argv) < 2:
    print('Usage:', sys.argv[0],
          ''' <params_file(s)> [redirect=0]
    params_file(s) :
        Parameters for the NeuralNet
        - params_file.py  : contains the initialization code
        - params_file.pkl : pickled file from a previous run (has wts too).
    redirect:
        1 - redirect stdout to a <SEED>.txt file
    ''')
    sys.exit()

prms_file_name = sys.argv[1]

##########################################  Import Parameters

if prms_file_name.endswith('.pkl'):
    with open(prms_file_name, 'rb') as f:
        params = pickle.load(f)
else:
    with open(prms_file_name, 'r') as f:
        params = ast.literal_eval(f.read())

layers = params['layers']
tr_prms = params['training_params']
try:
    allwts = params['allwts']
except KeyError:
    allwts = None

## Init SEED
if (not 'SEED' in tr_prms) or (tr_prms['SEED'] is None):
    tr_prms['SEED'] = np.random.randint(0, 1e6)

out_file_head = os.path.splitext(os.path.basename(prms_file_name))[0]
if not prms_file_name.endswith('.pkl'):
    out_file_head += "_{:06d}".format(tr_prms['SEED'])

if sys.argv[-1] is '1':
    print("Printing output to {}.txt".format(out_file_head), file=sys.stderr)
    sys.stdout = WrapOut(True, out_file_head + '.txt')
else:
    sys.stdout = WrapOut(False)


##########################################  Print Parameters

print(' '.join(sys.argv), file=sys.stderr)
print(' '.join(sys.argv))
start_time = datetime.now()
print('Time   :', start_time.strftime('%Y-%m-%d %H:%M:%S'))
print('Device : {} ({})'.format(th.config.device, th.config.floatX))
print('Host   :', socket.gethostname())

print(nn.get_layers_info(layers))
print(nn.get_training_params_info(tr_prms))

##########################################  Load Data

print("\nLoading the data ...")

try:
    img_sz = layers[0][1]["img_sz"]
except KeyError:
    img_sz = layers[0][1]["img_sz"] = 32

pad_width = (img_sz-32)//2
trin_x, trin_y = utils.load_pad_info("train", pad_width, 0)
test_x, test_y = utils.load_pad_info("test", pad_width, 0)

batch_sz = tr_prms['BATCH_SZ']
n_train = len(trin_y)
n_test = len(test_y)

trin_x = share(trin_x)
test_x = share(test_x)
trin_y = share(trin_y, 'int32')
test_y = share(test_y, 'int32')

################################################
print("\nInitializing the net ... ")
net = nn.NeuralNet(layers, tr_prms, allwts)
print(net)
print(net.get_wts_info(detailed=True).replace("\n\t", ""))

print("\nCompiling ... ")
training_fn = net.get_trin_model(trin_x, trin_y, take_index_list=True)
test_fn_tr = net.get_test_model(trin_x, trin_y)
test_fn_te = net.get_test_model(test_x, test_y)

tr_corpus_sz = n_train
te_corpus_sz = n_test
nEpochs = tr_prms['NUM_EPOCHS']
nTrBatches = tr_corpus_sz // batch_sz
nTeBatches = te_corpus_sz // batch_sz

############################################## MORE HELPERS 


def test_wrapper(nylist):
    sym_err, pr_true, n = 0., 0., 0
    for sym_err_, pr_true_ in nylist:
        sym_err += sym_err_
        pr_true += pr_true_
        n += 1
    return 100 * sym_err / n, 100 * pr_true / n

aux_err_name = 'P(MLE)'


def get_test_indices(tot_samps, bth_samps=tr_prms['TEST_SAMP_SZ']):
    n_bths_each = int(bth_samps / batch_sz)
    n_bths_all = int(tot_samps / batch_sz)
    cur = 0
    while True:
        yield [i % n_bths_all for i in range(cur, cur + n_bths_each)]
        cur = (cur + n_bths_each) % n_bths_all


test_indices = get_test_indices(te_corpus_sz)
trin_indices = get_test_indices(tr_corpus_sz)
pickle_file_name = out_file_head + '_{:02.0f}.pkl'
saved_file_name = None


def do_test():
    global saved_file_name
    test_err, aux_test_err = test_wrapper(test_fn_te(i)
                                          for i in next(test_indices))
    trin_err, aux_trin_err = test_wrapper(test_fn_tr(i)
                                          for i in next(trin_indices))
    print("{:5.2f}%  ({:5.2f}%)      {:5.2f}%  ({:5.2f}%)".format(
        trin_err, aux_trin_err, test_err, aux_test_err))
    sys.stdout.forceflush()

    if saved_file_name:
        os.remove(saved_file_name)

    saved_file_name = pickle_file_name.format(test_err)
    with open(saved_file_name, 'wb') as pkl_file:
        pickle.dump(net.get_init_params(), pkl_file, -1)

############################################ Training Loop
shuffled_indices = np.arange(tr_corpus_sz).astype("int32")

print("Training ...")
print("Epoch   Cost  Tr_Error Tr_{0}    Te_Error Te_{0}".format(aux_err_name))
for epoch in range(nEpochs):
    total_cost = 0
    np.random.shuffle(shuffled_indices)

    for ibatch in range(nTrBatches):
        batch = shuffled_indices[ibatch * batch_sz:(ibatch + 1) * batch_sz]
        output = training_fn(batch)
        total_cost += output[0]

        if np.isnan(total_cost):
            print("Epoch:{} Iteration:{}".format(epoch, ibatch))
            print(net.get_wts_info(detailed=True))
            raise ZeroDivisionError("Nan cost at Epoch:{} Iteration:{}"
                                    "".format(epoch, ibatch))

    if epoch % tr_prms['EPOCHS_TO_TEST'] == 0:
        print("{:3d} {:>8.2f}".format(net.get_epoch(), total_cost), end='    ')
        do_test()
        if total_cost > 1e9:
            print(net.get_wts_info(detailed=True))

    if epoch % tr_prms['EPOCHS_TO_RESET_GRADIENT'] == 0:
        net.reset_accumulated_gradients()

    net.inc_epoch_set_rate()

########################################## Final Error Rates

test_err, aux_test_err = test_wrapper(test_fn_te(i)
                                      for i in range(te_corpus_sz//batch_sz))
trin_err, aux_trin_err = test_wrapper(test_fn_tr(i)
                                      for i in range(tr_corpus_sz//batch_sz))

end_time = datetime.now()
tott = end_time-start_time
secs = tott.seconds
hours, minutes = tott.days*24 + tott.seconds//3600, (tott.seconds//60)%60
secs %= 60
print('Time   : {:%Y-%m-%d %H:%M:%S}\tRun-time: {}:{:02d}:{:02d}'.format(
    end_time, hours, minutes, secs))
print("{:3d} {:>8.2f}".format(net.get_epoch(), 0), end='    ')
print("{:5.2f}%  ({:5.2f}%)      {:5.2f}%  ({:5.2f}%)".format(
        trin_err, aux_trin_err, test_err, aux_test_err))
