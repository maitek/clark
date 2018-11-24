import numpy as np
import pyopencl as cl

ctx = cl.create_some_context(interactive=False,answers=[0,1])
queue = cl.CommandQueue(ctx)


class Tensor():
    def __init__(self,data):
        self.data = data
        self.prg = cl.Program(ctx, """
            __kernel void add(
                __global const float *a_g, __global const float *b_g, __global float *res_g)
            {
              int gid = get_global_id(0);
              res_g[gid] = a_g[gid] + b_g[gid];
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
        self.prg.add(queue, self.data.shape, None, a_g, b_g, res_g)

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
    a = ones((50,10),np.float32)
    b = ones((50,10),np.float32)

    c = a + b
    print(a.data, " + ", b.data, " = ", c.data)



