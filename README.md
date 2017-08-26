# Fashion-MNIST

[![Gitter](https://badges.gitter.im/zalandoresearch/fashion-mnist.svg)](https://gitter.im/fashion-mnist/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link)
[![Readme-CN](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87%E6%96%87%E6%A1%A3-blue.svg)](README.zh-CN.md)

A dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. `Fashion-MNIST` is intended to serve as a direct **drop-in replacement** of the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for benchmarking machine learning algorithms.

Here is an example how the data looks like (*each class takes three-rows*):

![](doc/img/fashion-mnist-sprite.png)

<img src="doc/img/embedding.gif" width="100%">

## Why?

The original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) contains a lot of handwritten digits. People from AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset they would try on. *"If it doesn't work on MNIST, it **won't work** at all"*, they said. *"Well, if it does work on MNIST, it may still fail on others."* 

`Fashion-MNIST` is intended to serve as a direct drop-in replacement for the original MNIST dataset to benchmark machine learning algorithms, as it shares the same image size and the structure of training and testing splits.

### To Serious Machine Learning Researchers

Seriously, we are talking about replacing MNIST. Here are some good reasons:

- MNIST is too easy. Check out [our side-by-side benchmark](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/) and ["Most pairs of MNIST digits can be distinguished pretty well by just one pixel"](https://gist.github.com/dgrtwo/aaef94ecc6a60cd50322c0054cc04478)
- MNIST is overused. Check out ["Ian Goodfellow wants people to move away from mnist"](https://twitter.com/goodfellow_ian/status/852591106655043584)
- MNIST can not represent modern CV tasks. Check out ["Ideas on MNIST do not transfer to real CV"](https://twitter.com/fchollet/status/852592598128615424)

## Get the Data

You can use direct links to download the the dataset. The data is stored in the **same** format as the original [MNIST data](http://yann.lecun.com/exdb/mnist/).

| Name  | Content | Examples | Size | Link
| --- | --- |--- | --- |--- |
| `train-images-idx3-ubyte.gz`  | training set images  | 60,000|26 MBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)|
| `train-labels-idx1-ubyte.gz`  | training set labels  |60,000|29 KBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz)|
| `t10k-images-idx3-ubyte.gz`  | test set images  | 10,000|4.2 MBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz)|
| `t10k-labels-idx1-ubyte.gz`  | test set labels  | 10,000| 5.0 KBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz)|

Or you can clone this repository, the dataset is under `data/fashion`. This repo contains some scripts for benchmark and visualization.
   
```bash
git clone git@github.com:zalandoresearch/fashion-mnist.git
```

### Labels
Each training and test example is assigned to one of the following labels:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Usage

### Loading data with Python (`numpy` is required)
- use `utils/mnist_reader` in this repo:
```python
import mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
```

### Loading data with Tensorflow
```python
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion')

data.train.next_batch(100)
```

### Loading data with other languages

As one of the most popular dataset in the Machine Learning community, people have implemented MNIST loader in many languages. They can be used to load `Fashion-MNIST` dataset as well (may require decompressing first). Note that they are not tested by us.

- [C](https://stackoverflow.com/a/10409376)
- [C++](https://github.com/wichtounet/mnist)
- [Java](https://stackoverflow.com/a/8301949)
- [Python](https://pypi.python.org/pypi/python-mnist) and [this](https://pypi.python.org/pypi/mnist)
- [Scala](http://mxnet.io/tutorials/scala/mnist.html)
- [Go](https://github.com/schuyler/neural-go/blob/master/mnist/mnist.go)
- [C#](https://jamesmccaffrey.wordpress.com/2013/11/23/reading-the-mnist-data-set-with-c/)
- [NodeJS](https://github.com/ApelSYN/mnist_dl) and [this](https://github.com/cazala/mnist)
- [Swift](https://github.com/simonlee2/MNISTKit)
- [R](https://gist.github.com/brendano/39760)
- [Matlab](http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset) and [this](https://de.mathworks.com/matlabcentral/fileexchange/27675-read-digits-and-labels-from-mnist-database?focused=5154133&tab=function)
- [Ruby](https://github.com/gbuesing/mnist-ruby-test/blob/master/train/mnist_loader.rb)


## Benchmark
We build an automatic benchmarking system based on `scikit-learn`, covering 125 classifiers with different parameters. [Results can be found here.](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/)

<img src="doc/img/benchmark.gif" width="100%">

You can reproduce the results by running `benchmark/runner.py`. A recommend way is to  build and deploy this docker container. 

You are welcome to submit your benchmark. Please create a new issue, your results will be listed here. Check out the [Contributing](https://github.com/zalandoresearch/fashion-mnist#contributing) section for details. Before submitting a benchmark, please make sure it is not listed [in this list](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/).  

| Classifier | Preprocessing | Test accuracy | Submitter| Code |
| --- | --- | --- | --- | --- |
|2 Conv Layers with max pooling (Keras) | None | 0.876 | [Kashif Rasul](https://twitter.com/krasul) | [zalando_mnist_cnn](https://gist.github.com/kashif/76792939dd6f473b7404474989cb62a8) |
|2 Conv Layers with max pooling (Tensorflow) | None | 0.916| [Tensorflow's doc](https://www.tensorflow.org/tutorials/layers) | [convnet](/benchmark/convnet.py)|
|Simple 2 layer convnet <100K parameter | None | 0.925 | [Martin Mundt](https://twitter.com/mundt_martin/status/901369943052210176) | [pytorch_tiny_custom_mnist_adam](https://github.com/hardmaru/pytorch_notebooks/blob/master/pytorch_tiny_custom_mnist_adam.ipynb)| 
|GRU+SVM | None| 0.888 | [@AFAgarap](https://github.com/AFAgarap) | [gru_svm_zalando_dropout](https://gist.githubusercontent.com/AFAgarap/92c1c4a5dd771999b0201ec0e7edfee0/raw/828fbda0e466dacb1fad66549e0e3022e1c7263a/gru_svm_zalando_dropout.py)|
|GRU+SVM with dropout | None| 0.855 | [@AFAgarap](https://github.com/AFAgarap) | [gru_svm_zalando_dropout](https://gist.githubusercontent.com/AFAgarap/92c1c4a5dd771999b0201ec0e7edfee0/raw/828fbda0e466dacb1fad66549e0e3022e1c7263a/gru_svm_zalando_dropout.py)|
|WRN40-4 8.9M params | standard preprocessing (mean/std subtraction/division) and augmentation (random crops/horizontal flips)| 0.967 | [@ajbrock](https://github.com/ajbrock) | :cry: [NA](https://github.com/zalandoresearch/fashion-mnist/issues/10) |
|DenseNet-BC 768K | standard preprocessing (mean/std subtraction/division) and augmentation (random crops/horizontal flips) | 0.954 | [@ajbrock](https://github.com/ajbrock)  | :cry: [NA](https://github.com/zalandoresearch/fashion-mnist/issues/10)|

### Other Explorations

#### Generative adversarial networks (GANs) 
- [Make  a ghost wardrobe using DCGAN](https://twitter.com/spaceLenny/status/901488938023403520)
- [fashion-mnist的gan玩具](http://kexue.fm/archives/4540/)


## Visualization

### t-SNE on Fashion-MNIST (left) and original MNIST (right) 
<img src="doc/img/34d72c08.png" width="50%"><img src="doc/img/01e0c4be.png" width="50%">

### PCA on Fashion-MNIST (left) and original MNIST (right) 
<img src="doc/img/f04ba662.png" width="50%"><img src="doc/img/4433f0e1.png" width="50%">


## Contributing

Thanks for your interest in contributing! There are many ways to contribute to this project. [Get started here!](/CONTRIBUTING.md) And please check these [open issues](https://github.com/zalandoresearch/fashion-mnist/issues) for specific tasks.

## Contact
For discussion on the dataset, please use [![Gitter](https://badges.gitter.im/zalandoresearch/fashion-mnist.svg)](https://gitter.im/fashion-mnist/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link)


## Citing Fashion-MNIST
If you use Fashion-MNIST in a scientific publication, we would appreciate references to the following paper:

> [Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.](doc/arxiv.pdf) Han Xiao, Kashif Rasul, 
Roland 
Vollgraf. arXiv: TBA

Bibtex entry:
```latex
TBA
```

The article is scheduled to be announced on arXiv at Mon, 28 Aug 2017 00:00:00 GMT.

## License

The MIT License (MIT) Copyright © [2017] Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
