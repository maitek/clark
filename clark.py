import numpy as np
import pyopencl as cl
from time import time

ctx = cl.create_some_context(interactive=False,answers=[0,1])
queue = cl.CommandQueue(ctx)

class Tensor():
    def __init__(self,data):
        self.data = data
        self.data_gpu = None

        self.prg = cl.Program(ctx, """
            __kernel void add(
            __global const float *a_g, __global const float *b_g, __global float *res_g)
            {
              int row = get_global_id(0);
              res_g[row] = a_g[row] + b_g[row];
            }

            __kernel void mul(
            __global const float *a_g, __global const float *b_g, __global float *res_g)
            {
              int row = get_global_id(0);
              res_g[row] = a_g[row] * b_g[row];
            }

            // First naive implementation
            __kernel void matmult(const int M, const int N, const int K,
                                  const __global float* A,
                                  const __global float* B,
                                  __global float* C) {
                
                // Thread identifiers
                const int row = get_global_id(0); // Row ID of C (0..M)
                const int col = get_global_id(1); // Col ID of C (0..N)
             
                // Compute a single element (loop over K)
                float acc = 0.0f;
                for (int k=0; k<K; k++) {
                    acc += A[k*M + row] * B[col*K + k];
                }
             
                // Store the result
                C[col*M + row] = acc;
            }



            """).build()

        @property
        def shape(self):
            return self.data.shape

    def numpy(self):
        return self.data

    def cpu(self):
        cl.enqueue_copy(queue, self.data, self.data_gpu)
        return self

    def gpu(self):
        mf = cl.mem_flags
        self.data_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.data)
        return self

    def __repr__(self):
        self.cpu()
        return "[ clark.Tensor; shape {}; dtype {}; min {}; max {} ]".format(
            self.data.shape,
            self.data.dtype,
            self.data.min(),
            self.data.max())

    def __str__(self):
        return self.__repr__()
        
    def __add__(self,other):

        result = Tensor(np.empty_like(self.data))
        
        mf = cl.mem_flags
        result.data_gpu = cl.Buffer(ctx, mf.READ_ONLY, self.data.nbytes)

        block_size = 100
        self.prg.add(queue, self.data.flatten().shape, (block_size,), self.data_gpu, other.data_gpu, result.data_gpu)

        cl.enqueue_copy(queue, result.data, result.data_gpu)
        return result

    def __mul__(self,other):

        result = Tensor(np.empty_like(self.data))
        
        mf = cl.mem_flags
        result.data_gpu = cl.Buffer(ctx, mf.READ_ONLY, self.data.nbytes)

        block_size = 100
        self.prg.mul(queue, self.data.flatten().shape, (block_size,), self.data_gpu, other.data_gpu, result.data_gpu)

        #cl.enqueue_copy(queue, result.data, res_g)
        return result

    def matmult(self,other):
        # TODO fix this

        result = Tensor(np.empty_like(self.data))
        
        mf = cl.mem_flags
        result.data_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, self.data.nbytes)
        #a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.data)
        #b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=other.data)
        
        block_size = 100
        N = np.int32(self.data.shape[0])
        K = np.int32(self.data.shape[1])
        M = np.int32(other.data.shape[1])
        self.prg.matmult(queue, self.data.flatten().shape,(block_size,),N,K,M,self.data_gpu, other.data_gpu, result.data_gpu)

        #cl.enqueue_copy(queue, result.data, res_g)
        return result

def from_numpy(ndarray):
    t = Tensor(ndarray.shape)
    t.data = ndarray

def ones(shape,dtype=np.float32):
    t = Tensor(data=np.ones(shape,dtype=np.float32))
    return t

def zeros(shape,dtype=np.float32):
    t = Tensor(data=np.zeros(shape,dtype=np.float32))
    return t

def randn(shape,dtype=np.float32):
    t = Tensor(data=np.randn(shape,dtype=np.float32))
    return t


if __name__ == "__main__":

    size = 1000
    num_runs = 10
    

    print("====== Clark (gpu) ======")
    a = ones((size,size),np.float32).gpu()
    b = ones((size,size),np.float32).gpu()
    c = zeros((size,size),np.float32).gpu()

    tic = time()
    
    for i in range(num_runs):
        c = a.matmult(b)
        print(".")
    print("Time total {} runs: {}".format(num_runs,time()-tic))
    
    print("====== Numpy (cpu) ======")
    a = np.ones((size,size),np.float32)
    b = np.ones((size,size),np.float32)
    c = np.zeros((size,size),np.float32)
    
    tic = time()
    for i in range(num_runs):
        c =np.matmul(a,b)
        print(".")
    print("Time total {} runs: {}".format(num_runs,time()-tic))
    

    

    

