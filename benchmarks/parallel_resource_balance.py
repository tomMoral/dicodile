import time
import threading
from queue import Queue


N_CPU_KEYS = ['n_jobs', 'n_workers', 'n_cpus']


class CallItem:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

        for k in N_CPU_KEYS:
            if k in kwargs:
                self.n_workers = kwargs[k]
                break

    def __call__(self):
        return self.func(*self.args, **self.kwargs)


def delayed(function):

    def delayed_function(*args, **kwargs):
        return CallItem(function, *args, **kwargs)

    return delayed_function


class ParallelResourceBalance:

    def __init__(self, max_workers):

        self.max_workers = max_workers
        self.out_queue = Queue()
        self.running_threads = []
        self.used_workers = 0
        self.next_args = None

    def __call__(self, list_call_items):
        return self.parallel_run(list_call_items)

    def parallel_run(self, list_call_items):
        self.used_workers = 0
        it_args = enumerate(list_call_items)
        results = []

        while self.load_balance(it_args):
            idx, res, n_workers = self.out_queue.get()
            self.used_workers -= n_workers
            results.append((idx, res))

        while self.running_threads:
            t = self.running_threads.pop()
            t.join()

        while not self.out_queue.empty():
            idx, res, n_workers = self.out_queue.get()
            self.used_workers -= n_workers
            results.append((idx, res))

        return [res for _, res in sorted(results)]

    def run_one(self, idx, call_item):
        try:
            res = call_item()
        except Exception as e:
            import traceback
            traceback.print_exc()
            res = e
        self.out_queue.put((idx, res, call_item.n_workers))

    def start_in_thread(self, idx, call_item):
        t = threading.Thread(target=self.run_one, args=(idx, call_item))
        t.start()
        self.running_threads.append(t)
        time.sleep(5)

    def load_balance(self, it_args):
        try:
            if self.next_args is None:
                self.next_args = next(it_args)

            idx, call_item = self.next_args
            while self.used_workers + call_item.n_workers <= self.max_workers:
                self.start_in_thread(idx, call_item)
                self.used_workers += call_item.n_workers
                self.next_args = idx, call_item = next(it_args)
            print(f"Using {self.used_workers} CPUs")
            return True
        except StopIteration:
            return False
