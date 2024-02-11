import time


def time_function(func):
    start = time.time()
    func()
    end = time.time()
    return end - start
