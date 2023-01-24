import time
from typing import Tuple


class StopWatch:
    def measure(self, callback: callable, params: list = []) -> Tuple:
        t1 = time.time()
        results = callback(*params)
        t2 = time.time()
        t = t2 - t1

        return t, results
