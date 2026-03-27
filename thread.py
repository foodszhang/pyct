# 可随时推出的thread
from threading import Thread, Event

class ExitableThread(Thread):
    """
    可以在外部直接退出的线程
    注意没有实现run方法!!!!!

    """

    def __init__(self, *args, **kwargs):
        super().__init__(target=self.wait_for_event, daemon=True)
        self._true_args = args
        self._true_kwargs = kwargs
        self._stop_event = Event()
        self._start_child_thread = Event()

    def wait_for_event(self):
        self._true_thread = Thread(daemon=True, *self._true_args, **self._true_kwargs)
        self._start_child_thread.set()
        self._stop_event.wait()
        print('exit')

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()

    def start(self):
        super().start()
        self._start_child_thread.wait()
        self._true_thread.start()

    def join(self, timeout=None):
        self._true_thread.join(timeout)
        super().join(timeout)
