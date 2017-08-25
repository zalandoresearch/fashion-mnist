import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from configs import DATA_DIR, LOG_DIR, VIS_DIR
from utils import mnist_reader
from utils.helper import get_sprite_image

X, Y = mnist_reader.load_mnist(path=DATA_DIR, kind='t10k')

labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
Y_str = np.array([labels[j] for j in Y])
np.savetxt('Xtest.tsv', X, fmt='%.6e', delimiter='\t')
np.savetxt('Ytest.tsv', Y_str, fmt='%s')

plt.imsave('zalando-mnist-sprite.png', get_sprite_image(X), cmap='gray')

exit()

embedding_var = tf.Variable(X, name='mnist_pixels')
# Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = VIS_DIR + 'Ytest.tsv'
embedding.sprite.image_path = VIS_DIR + 'zalando-mnist-sprite.png'
# Specify the width and height of a single thumbnail.
embedding.sprite.single_image_dim.extend([28, 28])

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR + 'visualization')

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, LOG_DIR + 'visualization/model.ckpt', 0)
