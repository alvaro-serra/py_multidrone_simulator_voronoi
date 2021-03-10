import asyncio
import numpy as np
import time

arrays = [np.array([i,i]) for i in range(100000)]


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

@background
def processInput_parallel(array):
    return array * array

auxt1 = time.time()

results = [processInput_parallel(array) for array in arrays]
print(results)

print("elapsed time:",time.time()-auxt1)



def processInput(array):
    return array * array
auxt2 = time.time()
results2 = [processInput(array) for array in arrays]
print(results2)
print("elapsed time:",time.time()-auxt2)
