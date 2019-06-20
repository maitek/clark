import clark
import numpy as np

def test_add_1():

    a = clark.ones((10,10),np.float32).gpu()
    b = clark.ones((10,10),np.float32).gpu()

    a_np = np.ones((10,10),np.float32)
    b_np = np.ones((10,10),np.float32)

    result = a + b
    result = result.cpu()
    assert np.linalg.norm(result.data - (a_np + b_np)) == 0

def test_add_2():
    a = clark.zeros((10,20),np.float32).gpu()
    b = clark.ones((10,20),np.float32).gpu()

    a_np = np.zeros((10,20),np.float32)
    b_np = np.ones((10,20),np.float32)

    result = a + b
    result = result.cpu()
    assert np.linalg.norm(result.data - (a_np + b_np)) == 0


def test_add_3():
    a = clark.ones((100),np.float32)
    b = clark.ones((100),np.float32)
    b.data *= (-0.5)

    a_np = np.ones((100),np.float32)
    b_np = np.ones((100),np.float32) * (-0.5)

    result = a.gpu() + b.gpu()
    result = result.cpu()
    assert np.linalg.norm(result.data - (a_np + b_np)) == 0


def test_mul_1():

    a = clark.ones((10,10),np.float32)
    b = clark.ones((10,10),np.float32)
    b.data *= 2

    a_np = np.ones((10,10),np.float32)
    b_np = np.ones((10,10),np.float32) * 2

    result = a.gpu() * b.gpu()
    result = result.cpu()
    assert np.linalg.norm(result.data - (a_np * b_np)) == 0


def test_matmult_1():

    #a = clark.rand((10,10),np.float32)
    #b = clark.rand((10,10),np.float32)
    a = clark.ones((10,10))
    b = clark.ones((10,10))
    
    a_np = a.numpy()
    b_np = b.numpy()

    result = a.gpu().matmul(b.gpu())
    result = result.cpu()

    import pdb; pdb.set_trace()
    assert np.linalg.norm(result.data - np.matmul(a_np,b_np)) == 0