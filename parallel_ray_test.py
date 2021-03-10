from joblib import Parallel, delayed
import numpy as np
import time
import ray

magnitude = 10000
arrays = [np.array([i,i]) for i in range(magnitude)]

@ray.remote
def processInput_parallel(array):
    return array * array

ray.init(local_mode=False)
auxt1 = time.time()
process_id = [processInput_parallel.remote(array) for array in arrays]
results = ray.get(process_id)
print(results)

print("elapsed time:",time.time()-auxt1)


#### Comparison with sequential ####
def processInput(array):
    return array * array
auxt2 = time.time()
results2 = [processInput(array) for array in arrays]
print(results2)
print("elapsed time:",time.time()-auxt2)

for i in range(10):
    auxt1 = time.time()
    process_id = [processInput_parallel.remote(array) for array in arrays]
    results = ray.get(process_id)
    print(results)

    print("elapsed time:",time.time()-auxt1)

auxt1 = time.time()
results = Parallel(n_jobs=-1)(delayed(processInput)(array) for array in arrays)
print(results)
print("elapsed time:",time.time()-auxt1)