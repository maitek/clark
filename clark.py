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
            __kernel void matmul2(const int M, const int N, const int K,
                                  const __global float* A,
                                  const __global float* B,
                                  __global float* C) {
                
                // Thread identifiers
                const int globalRow = get_global_id(0); // Row ID of C (0..M)
                const int globalCol = get_global_id(1); // Col ID of C (0..N)
             
                // Compute a single element (loop over K)
                float acc = 0.0f;
                for (int k=0; k<K; k++) {
                    acc += A[k*M + globalRow] * B[globalCol*K + k];
                }
             
                // Store the result
                C[globalCol*M + globalRow] = acc;
            }

            __kernel void
            matmul(__global float* C, 
                      __global float* A, 
                      __global float* B, 
                      int wA, int wB)
            {
              
               int tx = get_global_id(0); 
               int ty = get_global_id(1);
             
               // value stores the element that is 
               // computed by the thread
               float value = 0;
               for (int k = 0; k < wA; ++k)
               {
                  float elementA = A[ty * wA + k];
                  float elementB = B[k * wB + tx];
                  value += elementA * elementB;
               }
             
               // Write the matrix to device memory each 
               // thread writes one element
               C[ty * wA + tx] = value;
            }

            """).build()

        @property
        def shape(self):
            return self.data.shape

    def numpy(self):
        return self.data

    def cpu(self):
        cl.enqueue_copy(queue, self.data, self.data_gpu) # TODO slow!!"
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

        return result

    def __mul__(self,other):

        result = Tensor(np.empty_like(self.data))
        
        mf = cl.mem_flags
        result.data_gpu = cl.Buffer(ctx, mf.READ_ONLY, self.data.nbytes)

        block_size = 100
        self.prg.mul(queue, self.data.flatten().shape, (block_size,), self.data_gpu, other.data_gpu, result.data_gpu)

        return result

    def matmul(self,other):
        # TODO fix this

        result = Tensor(np.empty_like(self.data))
        
        mf = cl.mem_flags
        result.data_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, self.data.nbytes)
        
        block_size = 100
        N = np.int32(self.data.shape[0])
        K = np.int32(self.data.shape[1])
        M = np.int32(other.data.shape[1])
        #self.prg.matmul(queue, self.data.flatten().shape,(block_size,),N,K,M,self.data_gpu, other.data_gpu, result.data_gpu)
        self.prg.matmul(queue, self.data.flatten().shape,(block_size,),result.data_gpu, self.data_gpu, other.data_gpu, N, M)

        #cl.enqueue_copy(queue, result.data, result.data_gpu)
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

def rand(shape,dtype=np.float32):
    args = [x for x in shape]
    t = Tensor(data=np.random.rand(*args).astype(dtype))
    return t

# TODO

# matmult not working correctly
# .cpu() extremely slow!
# move kernels to other file(s)
# matmult api should be clark.matmul(a,b) so that is similar to numpy
# perhaps I could use https://github.com/CNugteren/CLBlast/tree/master/src/pyclblast
# better test using hypothesis
