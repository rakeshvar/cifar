import pickle
import sys

if len(sys.argv) < 3:
    print("Usage: {} file.pkl num_epochs".format(sys.argv[0]))

prms_file_name = sys.argv[1]
with open(prms_file_name, 'rb') as f:
    params = pickle.load(f)

trp = params['training_params']

def printrp(trp_, info):
    print(info, " training params:")
    for k, v in trp_.items():
        print("\t", k, ":", v)
    print("   Current Learning Rate:", trp_['INIT_LEARNING_RATE'] /
            (1 + trp_['CUR_EPOCH'] / trp_['EPOCHS_TO_HALF_RATE']))

printrp(trp, "Before")

trp['CUR_EPOCH'] = 0
trp['NUM_EPOCHS'] = int(sys.argv[2])

printrp(trp, "After")
out_file_name = prms_file_name[:-3] + 'num.pkl'
with open(out_file_name, 'wb') as pkl_file:
    pickle.dump(params, pkl_file, -1)
print("Saved to:", out_file_name, end="\n\n")
