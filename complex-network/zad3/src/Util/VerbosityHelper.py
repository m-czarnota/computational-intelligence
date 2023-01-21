import time


class VerbosityHelper:
    def __init__(self, is_enabled: bool = True, verbosity_level: int = 1):
        self.is_enabled: bool = is_enabled
        self.verbosity_level: int = verbosity_level

    def verbose(self, callback: callable, params: list = [], required_level: int = 1, message: str = 'Time of execution'):
        if self.is_enabled is False or self.verbosity_level < required_level:
            return callback(*params)

        t1 = time.time()
        result = callback(*params)
        t2 = time.time()

        time_result = t2 - t1
        indents = ''.join(['\t' for _ in range(required_level)])
        print(f'{indents}{message} - {time_result:.4f}s')

        return result
