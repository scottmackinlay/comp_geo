"""
How do you use a determinant to tell which side of a line a point is on?
"""



import numpy as np
from matplotlib import pyplot as plt
import unittest

def reflect(a, b):
    """Reflect a across vector b
    """
    return 2 * proj(a,b) - a

def proj(a,b):
    """Projects a onto b
    """
    return np.dot(a,b) * b / (np.linalg.norm(b)**2)



class Test(unittest.TestCase):
    def test_proj(self):
        a = np.asarray([1,0])
        b = np.asarray([1,1])
        self.assertTrue(np.allclose(proj(a,b),np.asarray([0.5, 0.5])))

    def test_reflect(self):
        a = np.asarray([0,-1])
        b = np.asarray([1,0])
        self.assertTrue(np.allclose(reflect(a,b),np.asarray([0,1])))

        a = np.asarray([1,0])
        b = np.asarray([1,1])
        self.assertTrue(np.allclose(reflect(a,b),np.asarray([0,1])))

        a = np.asarray([9.8,10.3])
        b = np.asarray([1,1])
        self.assertTrue(np.allclose(reflect(a,b),np.asarray([10.3, 9.8])))

    def test_det(self):
        for _ in range(100):
            mat = np.random.rand(2, 2)
            a = mat[0, :]
            b = mat[1, :]

            refl_b = reflect(a, b)
            refl_mat = np.stack([a, refl_b], axis = 0)
            self.assertTrue(np.linalg.det(mat), -np.linalg.det(refl_mat))


if __name__ == "__main__":
    unittest.main()
