# Fashion-MNIST

[![GitHub stars](https://img.shields.io/github/stars/zalandoresearch/fashion-mnist.svg?style=flat&label=Star)](https://github.com/zalandoresearch/fashion-mnist/)
[![Gitter](https://badges.gitter.im/zalandoresearch/fashion-mnist.svg)](https://gitter.im/fashion-mnist/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link)
[![Readme-CN](https://img.shields.io/badge/README-中文-green.svg)](README.zh-CN.md)
[![Readme-JA](https://img.shields.io/badge/README-日本語-green.svg)](README.ja.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Year-In-Review](https://img.shields.io/badge/%F0%9F%8E%82-Year%20in%20Review-orange.svg)](https://hanxiao.github.io/2018/09/28/Fashion-MNIST-Year-In-Review/)

<details><summary>Table of Contents</summary><p>

* [Why we made Fashion-MNIST](#why-we-made-fashion-mnist)
* [Get the Data](#get-the-data)
* [Usage](#usage)
* [Benchmark](#benchmark)
* [Visualization](#visualization)
* [Contributing](#contributing)
* [Contact](#contact)
* [Citing Fashion-MNIST](#citing-fashion-mnist)
* [License](#license)
</p></details><p></p>


`Fashion-MNIST` is a dataset of [Zalando](https://jobs.zalando.com/tech/)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend `Fashion-MNIST` to serve as a direct **drop-in replacement** for the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Here's an example of how the data looks (*each class takes three-rows*):

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

[Many ML libraries](#loading-data-with-other-machine-learning-libraries) already include Fashion-MNIST data/API, give it a try!

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

Note, Tensorflow supports passing in a source url to the `read_data_sets`. You may use: 
```python
data = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
```

Also, an official Tensorflow tutorial of using `tf.keras`, a high-level API to train Fashion-MNIST [can be found here](https://www.tensorflow.org/tutorials/keras/classification).

### Loading data with other machine learning libraries 
To date, the following libraries have included `Fashion-MNIST` as a built-in dataset. Therefore, you don't need to download `Fashion-MNIST` by yourself. Just follow their API and you are ready to go.

- [Activeloop Hub](https://docs.activeloop.ai/datasets/fashion-mnist-dataset)
- [Apache MXNet Gluon](https://mxnet.apache.org/api/python/docs/api/gluon/data/vision/datasets/index.html#mxnet.gluon.data.vision.datasets.FashionMNIST)
- [TensorFlow.js](https://github.com/tensorflow/tfjs-examples/blob/master/fashion-mnist-vae/data.js)
- [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist)
- [Pytorch](https://pytorch.org/vision/stable/datasets.html#fashion-mnist)
- [Keras](https://keras.io/api/datasets/fashion_mnist/)
- [Edward](http://edwardlib.org/api/observations/fashion_mnist)
- [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
- [Torch](https://github.com/mingloo/fashion-mnist)
- [JuliaML](https://juliaml.github.io/MLDatasets.jl/latest/datasets/FashionMNIST/)
- [Chainer](https://docs.chainer.org/en/stable/reference/generated/chainer.datasets.get_fashion_mnist.html)
- [HuggingFace Datasets](https://huggingface.co/datasets/fashion_mnist)
 
You are welcome to make pull requests to other open-source machine learning packages, improving their support to `Fashion-MNIST` dataset.

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
- [Matlab](http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset)
- [Ruby](https://github.com/gbuesing/mnist-ruby-test/blob/master/train/mnist_loader.rb)
- [Rust](https://github.com/AtheMathmo/vision-rs/blob/master/src/fashion_mnist.rs)


## Benchmark
We built an automatic benchmarking system based on `scikit-learn` that covers 129 classifiers (but no deep learning) with different parameters. [Find the results here](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/).

<img src="doc/img/benchmark.gif" width="100%">

You can reproduce the results by running `benchmark/runner.py`. We recommend building and deploying [this Dockerfile](Dockerfile). 

You are welcome to submit your benchmark; simply create a new issue and we'll list your results here. Before doing that, please make sure it does not already appear [in this list](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/). Visit our [contributor guidelines](https://github.com/zalandoresearch/fashion-mnist#contributing) for additional details.

The table below collects the submitted benchmarks. Note that **we haven't yet tested these results**. You are welcome to validate the results using the code provided by the submitter. Test accuracy may differ due to the number of epoch, batch size, etc. To correct this table, please create a new issue.

| Classifier | Preprocessing | Fashion test accuracy | MNIST test accuracy | Submitter| Code |
| --- | --- | --- | --- | --- |--- |
|2 Conv+pooling | None | 0.876 | - | [Kashif Rasul](https://twitter.com/krasul) | [:link:](https://gist.github.com/kashif/76792939dd6f473b7404474989cb62a8) |
|2 Conv+pooling | None | 0.916| - |[Tensorflow's doc](https://www.tensorflow.org/tutorials/layers) | [:link:](/benchmark/convnet.py)|
|2 Conv+pooling+ELU activation (PyTorch)| None| 0.903| - | [@AbhirajHinge](https://github.com/AbhirajHinge) | [:link:](https://github.com/AbhirajHinge/CNN-with-Fashion-MNIST-dataset)|
|2 Conv | Normalization, random horizontal flip, random vertical flip, random translation, random rotation. | 0.919 |0.971 | [Kyriakos Efthymiadis](https://github.com/kefth)| [:link:](https://github.com/kefth/fashion-mnist)|
|2 Conv <100K parameters | None | 0.925 | 0.992 |[@hardmaru](https://twitter.com/hardmaru) | [:link:](https://github.com/hardmaru/pytorch_notebooks/blob/master/pytorch_tiny_custom_mnist_adam.ipynb)|
|2 Conv ~113K parameters | Normalization | 0.922| 0.993 |[Abel G.](https://github.com/abelusha) | [:link:](https://github.com/abelusha/MNIST-Fashion-CNN/blob/master/Fashon_MNIST_CNN_using_Keras_10_Runs.ipynb)|
|2 Conv+3 FC ~1.8M parameters| Normalization | 0.932 | 0.994 | [@Xfan1025](https://github.com/Xfan1025) |[:link:](https://github.com/Xfan1025/Fashion-MNIST/blob/master/fashion-mnist.ipynb) |
|2 Conv+3 FC ~500K parameters | Augmentation, batch normalization | 0.934 | 0.994 | [@cmasch](https://github.com/cmasch) |[:link:](https://github.com/cmasch/zalando-fashion-mnist) |
|2 Conv+pooling+BN | None | 0.934 | - | [@khanguyen1207](https://github.com/khanguyen1207) | [:link:](https://github.com/khanguyen1207/My-Machine-Learning-Corner/blob/master/Zalando%20MNIST/fashion.ipynb)|
|2 Conv+2 FC| Random Horizontal Flips|  0.939| -| [@ashmeet13](https://github.com/ashmeet13)|[:link:](https://github.com/ashmeet13/FashionMNIST-CNN)|
|3 Conv+2 FC | None | 0.907 | - | [@Cenk Bircanoğlu](https://github.com/cenkbircanoglu) | [:link:](https://github.com/cenkbircanoglu/openface/tree/master/fashion_mnist)|
|3 Conv+pooling+BN | None | 0.903 | 0.994 | [@meghanabhange](https://github.com/meghanabhange) | [:link:](https://github.com/meghanabhange/FashionMNIST-3-Layer-CNN) |
|3 Conv+pooling+2 FC+dropout | None | 0.926 | - | [@Umberto Griffo](https://github.com/umbertogriffo) | [:link:](https://github.com/umbertogriffo/Fashion-mnist-cnn-keras)|
|3 Conv+BN+pooling|None|0.921|0.992|[@gchhablani](https://github.com/gchhablani)|[:link:](https://github.com/gchhablani/CNN-with-FashionMNIST)| 
|5 Conv+BN+pooling|None|0.931|-|[@Noumanmufc1](https://github.com/Noumanmufc1)|[:link:](https://gist.github.com/Noumanmufc1/60f00e434f0ce42b6f4826029737490a)| 
|CNN with optional shortcuts, dense-like connectivity| standardization+augmentation+random erasing | 0.947 |-| [@kennivich](https://github.com/Dezhic) | [:link:](https://github.com/Dezhic/fashion-classifier)|
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
|MLP 256-128-100| None | 0.8833| - | [@heitorrapela](https://github.com/heitorrapela)| [:link:](https://github.com/heitorrapela/fashion-mnist-mlp)| 
|VGG16 26M parameters | None | 0.935| - | [@QuantumLiu](https://github.com/QuantumLiu)|[:link:](https://github.com/QuantumLiu/fashion-mnist-demo-by-Keras) [:link:](https://zhuanlan.zhihu.com/p/28968219)|
|WRN-28-10| standard preprocessing (mean/std subtraction/division) and augmentation (random crops/horizontal flips) | 0.959 | -| [@zhunzhong07](https://github.com/zhunzhong07)|[:link:](https://github.com/zhunzhong07/Random-Erasing)|
|WRN-28-10 + Random Erasing| standard preprocessing (mean/std subtraction/division) and augmentation (random crops/horizontal flips) | 0.963 | -| [@zhunzhong07](https://github.com/zhunzhong07)|[:link:](https://github.com/zhunzhong07/Random-Erasing)|
|Human Performance| Crowd-sourced evaluation of human (with no fashion expertise) performance. 1000 randomly sampled test images, 3 labels per image, majority labelling. | 0.835 | - | Leo  | - |
|Capsule Network 8M parameters| Normalization and shift at most 2 pixel and horizontal flip | 0.936 | - | [@XifengGuo](https://github.com/XifengGuo)  | [:link:](https://github.com/XifengGuo/CapsNet-Fashion-MNIST)|
|HOG+SVM| HOG | 0.926 | - | [@subalde](https://github.com/subalde) | [:link:](https://github.com/subalde/fashion-mnist)|
|XgBoost| scaling the pixel values to mean=0.0 and var=1.0| 0.898| 0.958| [@anktplwl91](https://github.com/anktplwl91)| [:link:](https://github.com/anktplwl91/fashion_mnist.git)|
|DENSER| - | 0.953| 0.997| [@fillassuncao](https://github.com/fillassuncao)| [:link:](https://github.com/fillassuncao/denser-models) [:link:](https://arxiv.org/pdf/1801.01563.pdf)|
|Dyra-Net| Rescale to unit interval | 0.906| -| [@Dirk Schäfer](https://github.com/disc5)| [:link:](https://github.com/disc5/dyra-net) [:link:](https://dl.acm.org/citation.cfm?id=3204176.3204200)|
|Google AutoML|24 compute hours (higher quality)| 0.939|-| [@Sebastian Heinz](https://github.com/sebastianheinz) |[:link:](https://www.statworx.com/de/blog/a-performance-benchmark-of-google-automl-vision-using-fashion-mnist/)|
|Fastai| Resnet50+Fine-tuning+Softmax on last layer's activations| 0.9312| - | [@Sayak](https://github.com/sayakpaul) | [:link:](https://github.com/sayakpaul/Experiments-on-Fashion-MNIST/)|


### Other Explorations of Fashion-MNIST

#### [Fashion-MNIST: Year in Review](https://hanxiao.github.io/2018/09/28/Fashion-MNIST-Year-In-Review/)
#### [Fashion-MNIST on Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=fashion-mnist&btnG=&oq=fas) 

#### Generative adversarial networks (GANs) 
- [Tensorflow implementation of various GANs and VAEs.](https://github.com/hwalsuklee/tensorflow-generative-model-collections) (**Recommend to read!** Note how various GANs generate different results on Fashion-MNIST, which can not be easily observed on the original MNIST.)
- [Make a ghost wardrobe using DCGAN](https://twitter.com/spaceLenny/status/901488938023403520)
- [fashion-mnist的gan玩具](http://kexue.fm/archives/4540/)
- [CGAN output after 5000 steps](https://github.com/a7b23/Conditional-GAN-using-tensorflow-slim)
- [GAN Playground - Explore Generative Adversarial Nets in your Browser](https://reiinakano.github.io/gan-playground/)

#### Clustering
- [Xifeng Guo's implementation](https://github.com/XifengGuo/DEC-keras) of [Unsupervised Deep Embedding for Clustering Analysis (DEC)](http://proceedings.mlr.press/v48/xieb16.pdf)
- [Leland McInnes's](https://github.com/lmcinnes) [Uniform Manifold Approximation and Projection (UMAP)](https://github.com/lmcinnes/umap)

#### Video Tutorial
*Machine Learning Meets Fashion* by Yufeng G @ Google Cloud

[![Machine Learning Meets Fashion](doc/img/ae143b2d.png)](https://youtu.be/RJudqel8DVA)

*Introduction to Kaggle Kernels* by [Yufeng G](https://twitter.com/yufengg) @ Google Cloud

[![Introduction to Kaggle Kernels](doc/img/853c717e.png)](https://youtu.be/FloMHMOU5Bs)

*动手学深度学习* by Mu Li @ Amazon AI

[![MXNet/Gluon中文频道](doc/img/e9514ab1.png)](https://youtu.be/kGktiYF5upk)

Apache MXNet으로 배워보는 딥러닝(Deep Learning) - 김무현 (AWS 솔루션즈아키텍트)

[![Apache MXNet으로 배워보는 딥러닝(Deep Learning)](doc/img/dd83f448.png)](https://youtu.be/H66GDuLsGl4)



## Visualization

### t-SNE on Fashion-MNIST (left) and original MNIST (right) 
<img src="doc/img/34d72c08.png" width="50%"><img src="doc/img/01e0c4be.png" width="50%">

### PCA on Fashion-MNIST (left) and original MNIST (right) 
<img src="doc/img/f04ba662.png" width="50%"><img src="doc/img/4433f0e1.png" width="50%">

### [UMAP](https://github.com/lmcinnes/umap) on Fashion-MNIST (left) and original MNIST (right) 
<img src="doc/img/umap_example_fashion_mnist1.png" width="50%"><img src="doc/img/umap_example_mnist1.png" width="50%">

### [PyMDE](https://github.com/cvxgrp/pymde) on Fashion-MNIST (left) and original MNIST (right) 
<img src="doc/img/pymde_example_fashion_mnist.png" width="50%"><img src="doc/img/pymde_example_mnist.png" width="50%">


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

[Who is citing Fashion-MNIST?](https://scholar.google.com/scholar?scisbd=2&q=%22fashion-mnist%22&hl=en&as_sdt=0,5) 

## License

The MIT License (MIT) Copyright © [2017] Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
