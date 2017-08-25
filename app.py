import threading

from benchmark.runner import JobManager
from configs import LOGGER
from utils.argparser import get_args_cli
from utils.helper import UploadS3Thread


def start_s3_sync():
    stop_flag = threading.Event()
    upload_s3_thread = UploadS3Thread(stop_flag)
    upload_s3_thread.start()


if __name__ == "__main__":
    arg_dict = get_args_cli()
    LOGGER.info('received task with args: %s' % arg_dict)
    start_s3_sync()
    jm = JobManager(**arg_dict)
    jm.start()
