from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import time

arrays = [np.array([i,i]) for i in range(10000)]
def processInput(array):
    return array * array

num_cores = multiprocessing.cpu_count()

auxt1 = time.time()
results = Parallel(n_jobs=-1)(delayed(processInput)(array) for array in arrays)
print(results)
print("elapsed time:",time.time()-auxt1)

auxt2 = time.time()
results2 = [processInput(array) for array in arrays]
print(results2)
print("elapsed time:",time.time()-auxt2)


for i in range(10):
    auxt1 = time.time()
    results = Parallel(n_jobs=-1)(delayed(processInput)(array) for array in arrays)
    print(results)
    print("elapsed time:",time.time()-auxt1)
