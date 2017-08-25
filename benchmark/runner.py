import itertools
import json
import random
import time
from ast import literal_eval as make_tuple
from multiprocessing import Process, Queue

import numpy as np
import psutil
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils import shuffle

from configs import LOGGER, DATA_DIR, BASELINE_PATH, JSON_LOGGER
from utils import mnist_reader
from utils.helper import now_int


class PredictJob:
    def __init__(self, clf_name, clf_par, num_repeat: int = 1):
        self.clf_name = clf_name
        self.clf_par = clf_par
        self.result = None
        self.start_time = None
        self.done_time = None
        self.num_repeat = num_repeat


class JobWorker(Process):
    def __init__(self, pending_q: Queue) -> None:
        super().__init__()
        self.pending_q = pending_q
        X, self.Y = mnist_reader.load_mnist(path=DATA_DIR, kind='train')
        Xt, self.Yt = mnist_reader.load_mnist(path=DATA_DIR, kind='t10k')
        scaler = preprocessing.StandardScaler().fit(X)
        self.X = scaler.transform(X)
        self.Xt = scaler.transform(Xt)
        # self.X = X[:100]
        # self.Y = self.Y[:100]

    def run(self) -> None:
        while True:
            cur_job = self.pending_q.get()  # type: PredictJob

            LOGGER.info('job received! repeat: %d classifier: "%s" parameter: "%s"' % (cur_job.num_repeat,
                                                                                       cur_job.clf_name,
                                                                                       cur_job.clf_par))
            if cur_job.clf_name in globals():
                try:
                    acc = []
                    cur_job.start_time = now_int()
                    for j in range(cur_job.num_repeat):
                        cur_score = self.get_accuracy(cur_job.clf_name, cur_job.clf_par, j)
                        acc.append(cur_score)
                        if len(acc) == 2 and abs(acc[0] - cur_score) < 1e-3:
                            LOGGER.info('%s is invariant to training data shuffling, will stop repeating!' %
                                        cur_job.clf_name)
                            break
                    cur_job.done_time = now_int()
                    test_info = {
                        'name': cur_job.clf_name,
                        'parameter': cur_job.clf_par,
                        'score': acc,
                        'start_time': cur_job.start_time,
                        'done_time': cur_job.done_time,
                        'num_repeat': len(acc),
                        'mean_accuracy': np.array(acc).mean(),
                        'std_accuracy': np.array(acc).std() * 2,
                        'time_per_repeat': int((cur_job.done_time - cur_job.start_time) / len(acc))
                    }

                    JSON_LOGGER.info(json.dumps(test_info, sort_keys=True))

                    LOGGER.info('done! acc: %0.3f (+/- %0.3f) repeated: %d classifier: "%s" '
                                'parameter: "%s" ' % (np.array(acc).mean(),
                                                      np.array(acc).std() * 2,
                                                      len(acc),
                                                      cur_job.clf_name,
                                                      cur_job.clf_par))
                except Exception as e:
                    LOGGER.error('%s with %s failed! reason: %s' % (cur_job.clf_name, cur_job.clf_par, e))
            else:
                LOGGER.error('Can not found "%s" in scikit-learn, missing import?' % cur_job.clf_name)

    def get_accuracy(self, clf_name, clf_par, id):
        start_time = time.clock()
        clf = globals()[clf_name](**clf_par)
        Xs, Ys = shuffle(self.X, self.Y)
        cur_score = clf.fit(Xs, Ys).score(self.Xt, self.Yt)
        duration = time.clock() - start_time
        LOGGER.info('#test: %d acc: %0.3f time: %.3fs classifier: "%s" parameter: "%s"' % (id, cur_score,
                                                                                           duration,
                                                                                           clf_name,
                                                                                           clf_par))
        return cur_score


class JobManager:
    def __init__(self, num_worker: int = 2, num_repeat: int = 2, do_shuffle: bool = False,
                 respawn_memory_pct: float = 90):
        self.pending_q = Queue()
        self.num_worker = num_worker
        self.num_repeat = num_repeat
        self.do_shuffle = do_shuffle
        self.valid_jobs = self._sanity_check(self._parse_tasks(BASELINE_PATH))
        self.respawn_memory_pct = respawn_memory_pct
        for v in self.valid_jobs:
            self.pending_q.put(v)

    def memory_guard(self):
        LOGGER.info('memory usage: %.1f%%, RESPAWN_LIMIT: %.1f%%',
                    psutil.virtual_memory()[2], self.respawn_memory_pct)
        if psutil.virtual_memory()[2] > self.respawn_memory_pct:
            LOGGER.warn('releasing memory now! kill iterator processes and restart!')
            self.restart()

    def restart(self):
        self.close()
        self.start()

    def _parse_list(self, v):
        for idx, vv in enumerate(v):
            if isinstance(vv, str) and vv.startswith('('):
                v[idx] = make_tuple(vv)
        return v

    def _parse_tasks(self, fn):
        with open(fn) as fp:
            tmp = json.load(fp)

        def get_par_comb(tmp, clf_name):
            all_par_vals = list(itertools.product(*[self._parse_list(vv)
                                                    for v in tmp['classifiers'][clf_name]
                                                    for vv in v.values()]))
            all_par_name = [vv for v in tmp['classifiers'][clf_name] for vv in v.keys()]
            return [{all_par_name[idx]: vv for idx, vv in enumerate(v)} for v in all_par_vals]

        result = [{v: vv} for v in tmp['classifiers'] for vv in get_par_comb(tmp, v)]
        for v in result:
            for vv in v.values():
                vv.update(tmp['common'])
        if self.do_shuffle:
            random.shuffle(result)
        return result

    def close(self):
        for w in self.workers:
            w.join(timeout=1)
            w.terminate()

    def start(self):
        self.workers = [JobWorker(self.pending_q) for _ in range(self.num_worker)]
        for w in self.workers:
            w.start()

    def _sanity_check(self, all_tasks):
        total_clf = 0
        failed_clf = 0
        Xt, Yt = mnist_reader.load_mnist(path=DATA_DIR, kind='t10k')
        Xt = preprocessing.StandardScaler().fit_transform(Xt)
        Xs, Ys = shuffle(Xt, Yt)
        num_dummy = 10
        Xs = Xs[:num_dummy]
        Ys = [j for j in range(10)]
        valid_jobs = []
        for v in all_tasks:
            clf_name = list(v.keys())[0]
            clf_par = list(v.values())[0]
            total_clf += 1
            try:
                globals()[clf_name](**clf_par).fit(Xs, Ys)
                valid_jobs.append(PredictJob(clf_name, clf_par, self.num_repeat))
            except Exception as e:
                failed_clf += 1
                LOGGER.error('Can not create classifier "%s" with parameter "%s". Reason: %s' % (clf_name, clf_par, e))
        LOGGER.info('%d classifiers to test, %d fail to create!' % (total_clf, failed_clf))
        return valid_jobs


# no use, just prevent intellij to remove implicit import
placeholder = [PassiveAggressiveClassifier,
               SGDClassifier,
               Perceptron,
               DecisionTreeClassifier,
               RandomForestClassifier,
               LogisticRegression,
               MLPClassifier,
               KNeighborsClassifier,
               SVC,
               GaussianNB,
               ExtraTreeClassifier,
               LinearSVC,
               GaussianProcessClassifier,
               GradientBoostingClassifier]

if __name__ == "__main__":
    # predicting()
    jm = JobManager()
    jm.start()
    # jm.start()
