import os
import subprocess
from datetime import datetime
from threading import Thread, Event

from configs import SYNC_INTERVAL, LOGGER, LOG_PATH, ROOT_DIR, SYNC_TIMEOUT, RESULT_PATH, SYNC_SCRIPT_PATH


def now_int():
    epoch = datetime.utcfromtimestamp(0)
    return int((datetime.now() - epoch).total_seconds())


class UploadS3Thread(Thread):
    def __init__(self, event: Event):
        super().__init__()
        self.stopped = event

    def run(self):
        while not self.stopped.wait(SYNC_INTERVAL):
            upload_result_s3()


def upload_result_s3():
    LOGGER.info("Syncing data to S3...")
    with open(LOG_PATH, 'a', 1) as logfile:
        proc = subprocess.Popen(
            "bash %s %s" % (SYNC_SCRIPT_PATH, RESULT_PATH),
            shell=True,
            stdin=subprocess.PIPE,
            stdout=logfile,
            stderr=logfile,
            cwd=ROOT_DIR,
            env=os.environ)

        # we have to wait until the training data is downloaded
        try:
            outs, errs = proc.communicate(timeout=SYNC_TIMEOUT)
            if outs:
                LOGGER.info(outs)
            if errs:
                LOGGER.error(errs)
        except subprocess.TimeoutExpired:
            proc.kill()


def create_sprite_image(images):
    import numpy as np
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def vector_to_matrix_mnist(mnist_digits):
    import numpy as np
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))


def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 255 - mnist_digits


def get_sprite_image(to_visualise, do_invert=True):
    to_visualise = vector_to_matrix_mnist(to_visualise)
    if do_invert:
        to_visualise = invert_grayscale(to_visualise)
    return create_sprite_image(to_visualise)
