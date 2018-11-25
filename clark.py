import numpy as np
import pyopencl as cl
from time import time

ctx = cl.create_some_context(interactive=False,answers=[0,1])
queue = cl.CommandQueue(ctx)

class Tensor():
    def __init__(self,data):
        self.data = data

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

    def __repr__(self):
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
        a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.data)
        b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=other.data)

        res_g = cl.Buffer(ctx, mf.WRITE_ONLY, self.data.nbytes)
        
        self.prg.add(queue, self.data.flatten().shape, None, a_g, b_g, res_g)

        cl.enqueue_copy(queue, result.data, res_g)
        return result

    def __mul__(self,other):

        result = Tensor(np.empty_like(self.data))
        
        mf = cl.mem_flags
        a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.data)
        b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=other.data)

        res_g = cl.Buffer(ctx, mf.WRITE_ONLY, self.data.nbytes)
        
        self.prg.mul(queue, self.data.flatten().shape, None, a_g, b_g, res_g)

        cl.enqueue_copy(queue, result.data, res_g)
        return result

    def matmult(self,other):
        # TODO fix this

        result = Tensor(np.empty_like(self.data))
        
        mf = cl.mem_flags
        a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.data)
        b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=other.data)

        res_g = cl.Buffer(ctx, mf.WRITE_ONLY, self.data.nbytes)
        
        self.prg.matmult(queue, self.data.flatten().shape, None, a_g, b_g, res_g)

        cl.enqueue_copy(queue, result.data, res_g)
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
    
    a = np.ones((10000,10000),np.float32)
    b = np.ones((10000,10000),np.float32)

    tic = time()
    c = a + b
    print(time()-tic)

    a = ones((10000,10000),np.float32)
    b = ones((10000,10000),np.float32)
    
    tic = time()
    c = a + b
    print(time()-tic)


    
    print(a.data, " + ", b.data, " = ", c.data)

    

