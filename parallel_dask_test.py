
import numpy as np
import time

from dask.distributed import Client

magnitude = 10000
arrays = [np.array([i,i]) for i in range(magnitude)]


def processInput_parallel(array):
    return array * array


client = Client(n_workers=12)
futures = []
auxt1 = time.time()

for i in range(magnitude):
    future = client.submit(processInput_parallel, arrays[i])
    futures.append(future)
results = client.gather(futures)
client.close()

print(results)

print("elapsed time:",time.time()-auxt1)


#### Comparison with sequential ####
def processInput(array):
    return array * array
auxt2 = time.time()
results2 = [processInput(array) for array in arrays]
print(results2)
print("elapsed time:",time.time()-auxt2)
