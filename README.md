# cifar
[Cifar 100](http://www.cs.toronto.edu/~kriz/cifar.html) is implemented in [theano](Theano/Theano) using the [theanet](rakeshvar/theanet) package. 
We have managed to get error rates like 35% on the test data, which is state-of-the-art according to 
[these benchmarks](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d313030). 


## Usage

### Training

```sh
python train.py params/0baby.prms [1]
```

This will train on the training data, (like for ever if you are using the bigger configurations) and 
output a pkl file with the parameters in it, that can be loaded later on. You also get a text file, 
if you specify the optional 1 at the end of the command. like 0baby_555555.txt. 
Here, we use 555555 as a random seed. This can be specified in the prms file.
You can edit the [theanet](rakeshvar/theanet) params file to your liking.

### Testing

```sh
python test.py 0baby_555555_80.pkl
```

## Dependencies

* You need to use Python 3.0 (Come on over now, in to the new century). 
* Numpy
* PIL
* [Theano](Theano/Theano)
* [Theanet](rakeshvar/theanet)
