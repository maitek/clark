import clark
from time import time
import numpy as np

if __name__ == "__main__":

    size = 1500
    num_runs = 5
    print("Matrix size {} x {}:".format(size,size))
    print("Number of runs {}".format(num_runs))

    print("====== Clark matmult (gpu) ======")
    
    tic = time()
    a = clark.rand((size,size),np.float32).gpu()
    b = clark.rand((size,size),np.float32).gpu()
    c = clark.zeros((size,size),np.float32).gpu()
    print("Time init: {}".format(time()-tic))

    for i in range(num_runs):
        c = a.matmul(b)
    print("Time total: {}".format(time()-tic))
    
    
    print("====== Numpy matmult (cpu) ======")
    tic = time()
    a = a.numpy()
    b = b.numpy()
    c = np.zeros((size,size),np.float32)
    print("Time init: {}".format(time()-tic))

    for i in range(num_runs):
        c =np.matmul(a,b)
    print("Time total {}".format(time()-tic))
    
