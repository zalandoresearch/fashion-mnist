# Fashion-MNIST

[![GitHub stars](https://img.shields.io/github/stars/zalandoresearch/fashion-mnist.svg?style=flat&label=Star)](https://github.com/zalandoresearch/fashion-mnist/)
[![Gitter](https://badges.gitter.im/zalandoresearch/fashion-mnist.svg)](https://gitter.im/fashion-mnist/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link)
[![Readme-EN](https://img.shields.io/badge/README-English-green.svg)](README.md)
[![Readme-CN](https://img.shields.io/badge/README-中文-green.svg)](README.zh-CN.md)


翻訳 : [(株)クラスキャット セールスインフォメーション](http://tensorflow.classcat.com/2017/08/28/tensorflow-fashion-mnist/)

60,000 サンプルの訓練セットと 10,000 サンプルのテストセットから成る、[Zalando](https://jobs.zalando.com/tech/) の記事の画像のデータセットです。各サンプルは 28×28 グレースケール画像で、10 クラスからのラベルと関連付けられています。`Fashion-MNIST` は、機械学習アルゴリズムのベンチマークのためのオリジナルの MNIST データセット の 直接的な差し込み式の (= drop-in) 置き換え としてサーブすることを意図しています。

ここにどのようにデータが見えるかのサンプルがあります (各クラスは３行取ります) :

![](doc/img/fashion-mnist-sprite.png)

<img src="doc/img/embedding.gif" width="100%">

## 何故でしょう？
   
オリジナルの [MNIST](http://yann.lecun.com/exdb/mnist/) データセットは沢山の手書き数字を含みます。AI/ML/データサイエンス・コミュニティの人々はこのデータセットを好みそして彼らのアルゴリズムを検証するためのベンチマークとしてそれを使用します。実際に、MNIST はしばしば試してみる最初のデータセットです。「もしそれが MNIST で動作しなければ、まったく動作しないだろう」と彼らは言いました。「そうですね～、もし MNIST で動作するとしても、他の上では依然として失敗するかもしれませんが。」
   
Fashion-MNIST は、機械学習アルゴリズムのベンチマークのためのオリジナルの MNIST データセットの直接的な差し込み式の (= drop-in) 置き換えとしてサーブすることを意図しています、というのはそれは同じ画像サイズでそして訓練及びテスト分割の構造を共有しているからです。

### 真面目な機械学習研究者へ

真面目な話し、MNIST を置き換えることについて話しをしています。幾つかの良い理由がここにあります :

- **MNIST は簡単過ぎます。** [私たちの比較ベンチマーク](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/) と “[Most pairs of MNIST digits can be distinguished pretty well by just one pixel](https://gist.github.com/dgrtwo/aaef94ecc6a60cd50322c0054cc04478)” を確かめてください。
- **MNIST は使用され過ぎています。** “[Ian Goodfellow wants people to move away from mnist.](https://twitter.com/goodfellow_ian/status/852591106655043584)”を確かめてください。
- **MNIST はモダンな CV タスクを表現できません。** “[François Cholle: Ideas on MNIST do not transfer to real CV.](https://twitter.com/fchollet/status/852594987527045120)” を確かめてください。

## データを取得する

データセットをダウンロードするためには直接リンクを使用することができます。データはオリジナルの [MNIST](http://yann.lecun.com/exdb/mnist/) データと同じフォーマットでストアされています。

| 名前  | 内容 | サンプル | サイズ | リンク | MD5チェックサム|
| --- | --- |--- | --- |--- |--- |
| `train-images-idx3-ubyte.gz`  | 訓練セット画像	  | 60,000|26 MBytes | [ダウンロード](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)|`8d4fb7e6c68d591d4c3dfef9ec88bf0d`|
| `train-labels-idx1-ubyte.gz`  | 訓練セット・ラベル |60,000|29 KBytes | [ダウンロード](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz)|`25c81989df183df01b3e8a0aad5dffbe`|
| `t10k-images-idx3-ubyte.gz`  | テストセット画像  | 10,000|4.3 MBytes | [ダウンロード](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz)|`bef4ecab320f06d8554ea6380940ec79`|
| `t10k-labels-idx1-ubyte.gz`  | テストセット・ラベル  | 10,000| 5.1 KBytes | [ダウンロード](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz)|`bb300cfdad3c16e7a12a480ee83cd310`|

あるいはこのレポジトリを clone することもできます、データセットは `data/fashion` の下です。この repo はベンチーマークと可視化のための幾つかのスクリプトを含みます。
   
```bash
git clone git@github.com:zalandoresearch/fashion-mnist.git
```

### ラベル
各訓練とテスト・サンプルは以下のラベル群の一つに割り当てられています :

| ラベル | 記述 |
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

## 使い方

### Python ([NumPy](http://www.numpy.org/)が必要)でデータをロードする

この repo の `utils/mnist_reader` を使用する :
```python
import mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
```

### でデータをロードする

[私たちのデータセットをダウンロードしてください](#データを取得する)ことを確認し、それを `data/fashion`の下に置きます。それ以外の場合、* Tensorflowは自動的に元のMNISTをダウンロードして使用します。 *

```python
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion')

data.train.next_batch(BATCH_SIZE)
```

### 他の機械学習ライブラリを使用する

今日まで、以下のライブラリは、組み込みデータセットとして `Fashion-MNIST`を含んでいます。 したがって、自分で`Fashion-MNIST`をダウンロードする必要はありません。 そのAPIに従うだけで、あなたは準備が整いました。

- [Apache MXNet Gluon (master ver.)](https://mxnet.incubator.apache.org/versions/master/api/python/gluon.html#vision)
- [deeplearn.js](https://pair-code.github.io/deeplearnjs/demos/model-builder/model-builder-demo.html)
- [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist)
- [Pytorch](https://github.com/pytorch/vision#mnist)
- [Keras](https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles)
- [Edward](http://edwardlib.org/api/observations/fashion_mnist)
- [Tensorflow (master ver.)](https://github.com/tensorflow/tensorflow/pull/12983)
- [Torch](https://github.com/mingloo/fashion-mnist)


ようこそ私たちに参加して、各機械学習ライブラリ用の`Fashion-MNIST`のサポートを追加してください。


### 他の言語でデータをロードする

機械学習コミュニティでもっとも人気のあるデータセットの一つですので、人々は多くの言語で MNIST loader を実装してきています。それらは `Fashion-MNIST` データセットをロードするためにも使用できるでしょう (最初に decompress する必要があるかもしれません)。それらは私たちによってテストはされていないことには注意してください。

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


## ベンチマーク
scikit-learn ベースの自動ベンチマーキング・システムを構築しました、これは異なるパラメータの 129 の (深層学習ではない) 分類器をカバーします。 [結果はここで見つかります。](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/).

<img src="doc/img/benchmark.gif" width="100%">

結果は benchmark/runner.py を実行することで再現できます。推奨方法はこの docker コンテナをビルドして deploy することです (訳注 : リンク欠落)。[this Dockerfile](Dockerfile). 

貴方のベンチマークを submit することを歓迎します。新しい issue を作成してください、貴方の結果はここでリストされます。詳細は [contributor guidelines](https://github.com/zalandoresearch/fashion-mnist#contributing) セクションを確認してください。ベンチマークを submit する前に、このリストにリストされていなことを必ず確認してください。

| 分類器 | 前処理	 | Fashion テスト精度| MNIST テスト精度 | Submitter| コード |
| --- | --- | --- | --- | --- |--- |
|2 Conv Layers with max pooling (Keras) | None | 0.876 | - | [Kashif Rasul](https://twitter.com/krasul) | [:link:](https://gist.github.com/kashif/76792939dd6f473b7404474989cb62a8) |
|2 Conv Layers with max pooling (Tensorflow) >300 epochs | None | 0.916| - |[Tensorflow's doc](https://www.tensorflow.org/tutorials/layers) | [:link:](/benchmark/convnet.py)|
|2 Conv Layers net | Normalization, random horizontal flip, random vertical flip, random translation, random rotation. | 0.919 |0.971 | [Kyriakos Efthymiadis](https://github.com/kefth)| [:link:](https://github.com/kefth/fashion-mnist)|
|2 Conv Layers net <100K parameters | None | 0.925 | 0.992 |[@hardmaru](https://twitter.com/hardmaru) | [:link:](https://github.com/hardmaru/pytorch_notebooks/blob/master/pytorch_tiny_custom_mnist_adam.ipynb)|
|2 Conv Layers with 3 FC 1.8M parameters| Normalization | 0.932 | 0.994 | [@Xfan1025](https://github.com/Xfan1025) |[:link:](https://github.com/Xfan1025/Fashion-MNIST/blob/master/fashion-mnist.ipynb) |
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


### 他の探求

#### Generative adversarial networks (GANs) 
- [Tensorflow implementation of various GANs and VAEs.](https://github.com/hwalsuklee/tensorflow-generative-model-collections) (**Recommend to read!** Note how various GANs generate different results on Fashion-MNIST, which can not be easily observed on the original MNIST.)
- [Make a ghost wardrobe using DCGAN](https://twitter.com/spaceLenny/status/901488938023403520)
- [fashion-mnist的gan玩具](http://kexue.fm/archives/4540/)
- [CGAN output after 5000 steps](https://github.com/a7b23/Conditional-GAN-using-tensorflow-slim)
- [live demo of Generative Adversarial Network model with deeplearn.js](http://cognitivechaos.com/playground/fashion-gan/)

## 可視化

### t-SNE on Fashion-MNIST (左) とオリジナルの MNIST (右)
<img src="doc/img/34d72c08.png" width="50%"><img src="doc/img/01e0c4be.png" width="50%">

### PCA on Fashion-MNIST (左) とオリジナルの MNIST (右)
<img src="doc/img/f04ba662.png" width="50%"><img src="doc/img/4433f0e1.png" width="50%">

## 貢献する

Thanks for your interest in contributing! There are many ways to get involved; start with our [contributor guidelines](/CONTRIBUTING.md) and then check these [open issues](https://github.com/zalandoresearch/fashion-mnist/issues) for specific tasks.

## 接触
To discuss the dataset, please use [![Gitter](https://badges.gitter.im/zalandoresearch/fashion-mnist.svg)](https://gitter.im/fashion-mnist/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link).

## 引用Fashion-MNIST
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
