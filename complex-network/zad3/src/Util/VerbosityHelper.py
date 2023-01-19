import time


class VerbosityHelper:
    @staticmethod
    def measure_time(callback: callable, message: str):
        t1 = time.time()
        result = callback()
        t2 = time.time()
        calc_time = t2 - t1

        print(f'{message} - {calc_time:.2f}')

        return result
