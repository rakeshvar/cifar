# cifar
[Cifar 100](http://www.cs.toronto.edu/~kriz/cifar.html) challenge is implemented via deep neural networks in [theano](Theano/Theano) using the [theanet](rakeshvar/theanet) package. This repository is the basis for the simulations shown in *Chapter 18* of the book **Computer Age Statistical Inference** by *Bradley Efron* and *Trevor Hastie*.

We have managed to get error rates as low as 35% on the test data, which ~~is~~ was state-of-the-art according to 
[these benchmarks](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d313030). To reproduce the 35% test error rate, you need to use the `55M.prms` parameter file, then train, retrain and retrain. 


## Usage

### Getting the Data

Get the [tar.gz file] (http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) from the [official site](http://www.cs.toronto.edu/~kriz/cifar.html) and unzip it to the root directory.
 
### Training

```sh
python train.py params/0baby.prms [1]     # For a test run with a small network
python train.py params/55M.prms [1]       # For a serious run with a huge network
```

This will train on the training data, (like for ever if you are using the bigger configurations) and 
output a `pkl` file with the parameters in it, that can be loaded later on. You also get a text file, 
if you specify the optional 1 at the end of the command, like `0baby_555555.txt`. 
Here `555555` is a random random-seed. (This can be specified in the prms file.)
You can edit the [theanet](rakeshvar/theanet) params file to your liking. 
Sample params files are given in the `params` folder.


#### Re-training

```sh
python train.py 55M_555555_42.pkl 
```

You could also give an existing `pkl` file as an input in stead of a `prms` 
file. This will initialize the network with those parameters and pick up the 
training from there. Although the learning rate could be low. The file 
`edit_pkl.py` will tell you how to edit the `pkl` files before re-training. 


### Testing

```sh
python test.py 55M_555555_42.re_37.re_35.pkl # This file has been obtained by triple retraining
```
This only opens the `test` dataset and saves the intermediate images as the input goes through the CNN. 

## Dependencies

* You need to use Python 3.0 (Come on over now, in to the new century). 
* Numpy
* PIL
* [Theano](Theano/Theano)
* [Theanet](rakeshvar/theanet)
