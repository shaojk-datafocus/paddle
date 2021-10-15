import time
from tqdm import tqdm


def generator():
    for i in range(5):
        yield i
        time.sleep(0.5)

bar = tqdm(generator())
for i in tqdm(bar,total=10):
    bar.postfix="这里%d~"%i