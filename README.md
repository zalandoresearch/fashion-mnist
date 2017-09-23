# Fashion-MNIST

[![GitHub stars](https://img.shields.io/github/stars/zalandoresearch/fashion-mnist.svg?style=flat&label=Star)](https://github.com/zalandoresearch/fashion-mnist/)
[![Gitter](https://badges.gitter.im/zalandoresearch/fashion-mnist.svg)](https://gitter.im/fashion-mnist/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link)
[![Readme-CN](https://img.shields.io/badge/README-中文-green.svg)](README.zh-CN.md)
[![Readme-JA](https://img.shields.io/badge/README-日本語-green.svg)](README.ja.md)


`Fashion-MNIST` is a dataset of [Zalando](https://jobs.zalando.com/tech/)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend `Fashion-MNIST` to serve as a direct **drop-in replacement** for the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Here's an example how the data looks (*each class takes three-rows*):

![](doc/img/fashion-mnist-sprite.png)

<img src="doc/img/embedding.gif" width="100%">

## Why we made Fashion-MNIST

The original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) contains a lot of handwritten digits. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset researchers try. *"If it doesn't work on MNIST, it **won't work** at all"*, they said. *"Well, if it does work on MNIST, it may still fail on others."* 

### To Serious Machine Learning Researchers

Seriously, we are talking about replacing MNIST. Here are some good reasons:

- **MNIST is too easy.** Convolutional nets can achieve 99.7% on MNIST. Classic machine learning algorithms can also achieve 97% easily. Check out [our side-by-side benchmark for Fashion-MNIST vs. MNIST](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/), and read "[Most pairs of MNIST digits can be distinguished pretty well by just one pixel](https://gist.github.com/dgrtwo/aaef94ecc6a60cd50322c0054cc04478)."
- **MNIST is overused.** In [this April 2017 Twitter thread](https://twitter.com/goodfellow_ian/status/852591106655043584), Google Brain research scientist and deep learning expert Ian Goodfellow calls for people to move away from MNIST.
- **MNIST can not represent modern CV tasks**, as noted in [this April 2017 Twitter thread](https://twitter.com/fchollet/status/852594987527045120), deep learning expert/Keras author François Chollet.

## Get the Data

You can use direct links to download the dataset. The data is stored in the **same** format as the original [MNIST data](http://yann.lecun.com/exdb/mnist/).

| Name  | Content | Examples | Size | Link | MD5 Checksum|
| --- | --- |--- | --- |--- |--- |
| `train-images-idx3-ubyte.gz`  | training set images  | 60,000|26 MBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)|`8d4fb7e6c68d591d4c3dfef9ec88bf0d`|
| `train-labels-idx1-ubyte.gz`  | training set labels  |60,000|29 KBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz)|`25c81989df183df01b3e8a0aad5dffbe`|
| `t10k-images-idx3-ubyte.gz`  | test set images  | 10,000|4.3 MBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz)|`bef4ecab320f06d8554ea6380940ec79`|
| `t10k-labels-idx1-ubyte.gz`  | test set labels  | 10,000| 5.1 KBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz)|`bb300cfdad3c16e7a12a480ee83cd310`|

Alternatively, you can clone this GitHub repository; the dataset appears under `data/fashion`. This repo also contains some scripts for benchmark and visualization.
   
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

### Loading data with Python (requires [NumPy](http://www.numpy.org/))

Use `utils/mnist_reader` in this repo:
```python
import mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
```

### Loading data with Tensorflow
Make sure you have [downloaded the data](#get-the-data) and placed it in `data/fashion`. Otherwise, *Tensorflow will download and use the original MNIST.*

```python
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion')

data.train.next_batch(BATCH_SIZE)
```

Note, Tensorflow (master ver.) supports passing in a source url to the `read_data_sets`. You may use: 
```python
data = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
```

### Loading data with other machine learning libraries 
To date, the following libraries have included `Fashion-MNIST` as a built-in dataset. Therefore, you don't need to download `Fashion-MNIST` by yourself. Just follow their API and you are ready to go.

- [Apache MXNet Gluon (master ver.)](https://mxnet.incubator.apache.org/versions/master/api/python/gluon.html#vision)
- [deeplearn.js](https://pair-code.github.io/deeplearnjs/demos/model-builder/model-builder-demo.html)
- [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist)
- [Pytorch](https://github.com/pytorch/vision#mnist)
- [Keras](https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles)
- [Edward](http://edwardlib.org/api/observations/fashion_mnist)
- [Tensorflow (master ver.)](https://github.com/tensorflow/tensorflow/pull/12983)
- [Torch](https://github.com/mingloo/fashion-mnist)

You are welcome to make pull requests to other open-source machine learning packages, improving their support on `Fashion-MNIST` dataset.

### Loading data with other languages

As one of the Machine Learning community's most popular datasets, MNIST has inspired people to implement loaders in many different languages. You can use these loaders with the `Fashion-MNIST` dataset as well. (Note: may require decompressing first.) To date, we haven't yet tested all of these loaders with Fashion-MNIST.

- [C](https://stackoverflow.com/a/10409376)
- [C++](https://github.com/wichtounet/mnist)
- [Java](https://stackoverflow.com/a/8301949)
- [Python](https://pypi.python.org/pypi/python-mnist) and [this](https://pypi.python.org/pypi/mnist)
- [Scala](http://mxnet.io/tutorials/scala/mnist.html)
- [Go](https://github.com/schuyler/neural-go/blob/master/mnist/mnist.go)
- [C#](https://jamesmccaffrey.wordpress.com/2013/11/23/reading-the-mnist-data-set-with-c/)
- [NodeJS](https://github.com/ApelSYN/mnist_dl) and [this](https://github.com/cazala/mnist)
- [Swift](https://github.com/simonlee2/MNISTKit)
- [R](https://gist.github.com/brendano/39760) and [this](https://github.com/maddin79/darch)
- [Matlab](http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset) and [this](https://de.mathworks.com/matlabcentral/fileexchange/27675-read-digits-and-labels-from-mnist-database?focused=5154133&tab=function)
- [Ruby](https://github.com/gbuesing/mnist-ruby-test/blob/master/train/mnist_loader.rb)


## Benchmark
We built an automatic benchmarking system based on `scikit-learn` that covers 129 classifiers (but no deep learning) with different parameters. [Find the results here](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/).

<img src="doc/img/benchmark.gif" width="100%">

You can reproduce the results by running `benchmark/runner.py`. We recommend building and deploying [this Dockerfile](Dockerfile). 

You are welcome to submit your benchmark; simply create a new issue and we'll list your results here. Before doing that, please make sure it does not already appear [in this list](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/). Visit our [contributor guidelines](https://github.com/zalandoresearch/fashion-mnist#contributing) for additional details.

The table below collects the submitted benchmarks. Note that **we haven't yet tested these results**. You are welcome to validate the results using the code provided by the submitter. Test accuracy may differ due to the number of epoch, batch size, etc. To correct this table, please create a new issue.

| Classifier | Preprocessing | Fashion test accuracy | MNIST test accuracy | Submitter| Code |
| --- | --- | --- | --- | --- |--- |
|2 Conv Layers with max pooling (Keras) | None | 0.876 | - | [Kashif Rasul](https://twitter.com/krasul) | [:link:](https://gist.github.com/kashif/76792939dd6f473b7404474989cb62a8) |
|2 Conv Layers with max pooling (Tensorflow) >300 epochs | None | 0.916| - |[Tensorflow's doc](https://www.tensorflow.org/tutorials/layers) | [:link:](/benchmark/convnet.py)|
|2 Conv Layers net | Normalization, random horizontal flip, random vertical flip, random translation, random rotation. | 0.919 |0.971 | [Kyriakos Efthymiadis](https://github.com/kefth)| [:link:](https://github.com/kefth/fashion-mnist)|
|2 Conv Layers net <100K parameters | None | 0.925 | 0.992 |[@hardmaru](https://twitter.com/hardmaru) | [:link:](https://github.com/hardmaru/pytorch_notebooks/blob/master/pytorch_tiny_custom_mnist_adam.ipynb)|
|2 Conv Layers with 3 FC 1.8M parameters | Normalization | 0.932 | 0.994 | [@Xfan1025](https://github.com/Xfan1025) |[:link:](https://github.com/Xfan1025/Fashion-MNIST/blob/master/fashion-mnist.ipynb) | 
|3 Conv layers and 2 FC | None | 0.907 | - | [@Cenk Bircanoğlu](https://github.com/cenkbircanoglu) | [:link:](https://github.com/cenkbircanoglu/openface/tree/master/fashion_mnist)|
|3 Conv+pooling and 2 FC+dropout | None | 0.926 | - | [@Umberto Griffo](https://github.com/umbertogriffo) | [:link:](https://github.com/umbertogriffo/Fashion-mnist-cnn-keras)|
|GRU+SVM | None| 0.888 | 0.965 | [@AFAgarap](https://github.com/AFAgarap) | [:link:](https://gist.githubusercontent.com/AFAgarap/92c1c4a5dd771999b0201ec0e7edfee0/raw/828fbda0e466dacb1fad66549e0e3022e1c7263a/gru_svm_zalando.py)|
|GRU+SVM with dropout | None| 0.897 | 0.988 | [@AFAgarap](https://github.com/AFAgarap) | [:link:](https://gist.githubusercontent.com/AFAgarap/92c1c4a5dd771999b0201ec0e7edfee0/raw/58dbe7cd8b0d83e4386cd6896766113b1a9af096/gru_svm_zalando_dropout.py)|
|WRN40-4 8.9M params | standard preprocessing (mean/std subtraction/division) and augmentation (random crops/horizontal flips)| 0.967 | - |[@ajbrock](https://github.com/ajbrock) | [:link:](https://github.com/xternalz/WideResNet-pytorch)  [:link:](https://github.com/ajbrock/FreezeOut) |
|DenseNet-BC 768K params| standard preprocessing (mean/std subtraction/division) and augmentation (random crops/horizontal flips) | 0.954 | - |[@ajbrock](https://github.com/ajbrock)  | [:link:](https://github.com/bamos/densenet.pytorch)  [:link:](https://github.com/ajbrock/FreezeOut) |
|MobileNet | augmentation (horizontal flips)| 0.950|- | [@苏剑林](https://github.com/bojone)| [:link:](http://kexue.fm/archives/4556/)|
|ResNet18 | Normalization, random horizontal flip, random vertical flip, random translation, random rotation. | 0.949 | 0.979 |[Kyriakos Efthymiadis](https://github.com/kefth)| [:link:](https://github.com/kefth/fashion-mnist)|
|GoogleNet with cross-entropy loss | None | 0.937 | - | [@Cenk Bircanoğlu](https://github.com/cenkbircanoglu) | [:link:](https://github.com/cenkbircanoglu/openface/tree/master/fashion_mnist)|
|AlexNet with Triplet loss| None | 0.899 | - | [@Cenk Bircanoğlu](https://github.com/cenkbircanoglu) | [:link:](https://github.com/cenkbircanoglu/openface/tree/master/fashion_mnist)|
|SqueezeNet with cyclical learning rate 200 epochs| None| 0.900| - | [@snakers4](https://github.com/snakers4) | [:link:](https://github.com/zalandoresearch/fashion-mnist/files/1263340/squeeze_net_mnist.zip)|
|Dual path network with wide resnet 28-10|standard preprocessing (mean/std subtraction/division) and augmentation (random crops/horizontal flips) |0.957|-|[@Queequeg](https://github.com/Queequeg92)|[:link:](https://github.com/Queequeg92/DualPathNet)|
|MLP 256-128-64| None | 0.900| - | [@lianghong](https://github.com/lianghong)| [:link:](https://github.com/lianghong/fashion_mnist-on-mxnet)| 
|VGG16 26M parameters | None | 0.935| - | [@QuantumLiu](https://github.com/QuantumLiu)|[:link:](https://github.com/QuantumLiu/fashion-mnist-demo-by-Keras) [:link:](https://zhuanlan.zhihu.com/p/28968219)|
|WRN-28-10| standard preprocessing (mean/std subtraction/division) and augmentation (random crops/horizontal flips) | 0.959 | -| [@zhunzhong07](https://github.com/zhunzhong07)|[:link:](https://github.com/zhunzhong07/Random-Erasing)|
|WRN-28-10 + Random Erasing| standard preprocessing (mean/std subtraction/division) and augmentation (random crops/horizontal flips) | 0.963 | -| [@zhunzhong07](https://github.com/zhunzhong07)|[:link:](https://github.com/zhunzhong07/Random-Erasing)|
|Human Performance| Crowd-Sourced evaluation of human performance. 1000 randomly sampled test images, 3 labels per image, majority labelling. | 0.835 | - | Leo  | - 

### Other Explorations of Fashion-MNIST

#### Generative adversarial networks (GANs) 
- [Tensorflow implementation of various GANs and VAEs.](https://github.com/hwalsuklee/tensorflow-generative-model-collections) (**Recommend to read!** Note how various GANs generate different results on Fashion-MNIST, which can not be easily observed on the original MNIST.)
- [Make a ghost wardrobe using DCGAN](https://twitter.com/spaceLenny/status/901488938023403520)
- [fashion-mnist的gan玩具](http://kexue.fm/archives/4540/)
- [CGAN output after 5000 steps](https://github.com/a7b23/Conditional-GAN-using-tensorflow-slim)
- [live demo of Generative Adversarial Network model with deeplearn.js](http://cognitivechaos.com/playground/fashion-gan/)


## Visualization

### t-SNE on Fashion-MNIST (left) and original MNIST (right) 
<img src="doc/img/34d72c08.png" width="50%"><img src="doc/img/01e0c4be.png" width="50%">

### PCA on Fashion-MNIST (left) and original MNIST (right) 
<img src="doc/img/f04ba662.png" width="50%"><img src="doc/img/4433f0e1.png" width="50%">

## Contributing

Thanks for your interest in contributing! There are many ways to get involved; start with our [contributor guidelines](/CONTRIBUTING.md) and then check these [open issues](https://github.com/zalandoresearch/fashion-mnist/issues) for specific tasks.

## Contact
To discuss the dataset, please use [![Gitter](https://badges.gitter.im/zalandoresearch/fashion-mnist.svg)](https://gitter.im/fashion-mnist/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link).

## Citing Fashion-MNIST
If you use Fashion-MNIST in a scientific publication, we would appreciate references to the following paper:

**Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. [arXiv:1708.07747](http://arxiv.org/abs/1708.07747)**

Biblatex entry:
```latex
@online{xiao2017/online,
  author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
  title        = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  date         = {2017-08-28},
  year         = {2017},
  eprintclass  = {cs.LG},
  eprinttype   = {arXiv},
  eprint       = {cs.LG/1708.07747},
}
```

## License

The MIT License (MIT) Copyright © [2017] Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
