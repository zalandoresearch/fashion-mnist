import os
from random import randint

APP_NAME = '%s-%d' % ('fashion-mnist', randint(0, 100))
LOG_FORMAT = '%(asctime)-15s %(filename)s:%(funcName)s:[%(levelname)s] %(message)s'
JSON_FORMAT = '%(message)s'

RUN_LOCALLY = False
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
TEST_DIR = ROOT_DIR + 'test/'
DATA_DIR = ROOT_DIR + 'data/fashion'
VIS_DIR = ROOT_DIR + 'visualization/'
MODEL_SAVE_DIR = ROOT_DIR + 'save/'
MULTI_TASK_MODEL = '20170814-153653'
TEST_DATA_DIR = TEST_DIR + 'data/'
LOG_DIR = ROOT_DIR + 'log/'
RESULT_DIR = ROOT_DIR + 'result/'
TEMPLATE_DIR = ROOT_DIR + 'templates/'
STATIC_DIR = ROOT_DIR + 'static/'
SCRIPT_DIR = ROOT_DIR + 'script/'
BASELINE_PATH = ROOT_DIR + 'benchmark/baselines.json'

Q2A_SUFFIX = '-merged-ad1-20170501+36D+20170605.json.gz'

SYNC_SCRIPT_PATH = SCRIPT_DIR + 'sync_s3.sh'
DOWNLOAD_SCRIPT_PATH = SCRIPT_DIR + 'load_s3_json.sh'
LOG_PATH = LOG_DIR + APP_NAME + '.log'
RESULT_PATH = RESULT_DIR + APP_NAME + '.json'

Q2A_PATH = DATA_DIR + "query2brand-train.tfr"
Q2A_INFO = DATA_DIR + "query2brand.json"
MAX_ITEM_PER_ATTRIBUTE = 20

LOSS_JITTER = 1e-4
SYNC_INTERVAL = 300.0  # sync every 5 minutes
SYNC_TIMEOUT = 600
FIRST_SYNC_DELAY = 300.0  # do the first task only after 5 minutes.

RNN_ARGS_JSON = ROOT_DIR + 'nn/queryclf/config.json'

Q2A_JSON_AKEY1 = 'attributes'
Q2A_JSON_AKEY2 = 'value'


def touch(fname: str, times=None, create_dirs: bool = False):
    if create_dirs:
        base_dir = os.path.dirname(fname)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    with open(fname, 'a'):
        os.utime(fname, times)


def touch_dir(base_dir: str) -> None:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


def _get_logger(name: str):
    import logging.handlers
    touch(LOG_PATH, create_dirs=True)
    touch_dir(MODEL_SAVE_DIR)
    l = logging.getLogger(name)
    l.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    l.addHandler(fh)
    l.addHandler(ch)
    return l


def get_json_logger(name: str):
    import logging.handlers
    touch(RESULT_PATH, create_dirs=True)
    l = logging.getLogger(__name__ + name)
    l.setLevel(logging.INFO)
    # add rotator to the logger. it's lazy in the sense that it wont rotate unless there are new logs
    fh = logging.FileHandler(RESULT_PATH)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(JSON_FORMAT))
    l.addHandler(fh)
    return l


LOGGER = _get_logger(__name__)
JSON_LOGGER = get_json_logger('json' + __name__)
