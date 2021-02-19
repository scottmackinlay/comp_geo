import time
import numpy as np
import unittest
from matplotlib import pyplot as plt


def check_right(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
    """Checks that p3 is to the right of the directed line from p1 to p2.

    Args:
        p1 (np.ndarray): The origin of the directed line.
        p2 (np.ndarray): The end point of the directed line.
        p3 (np.ndarray): The point to check if it is to the right of the line.

    Returns:
        bool: True if the point is to the right, False otherwise.
    """
    mat = np.asarray([[1, p1[0], p1[1]],
                      [1, p2[0], p2[1]],
                      [1, p3[0], p3[1]]])

    return np.linalg.det(mat) < 0


def graham_scan(points):
    sort_pts = points[np.lexsort((points[:, 1], points[:, 0]), axis=0)]
    res_hull_top = [*sort_pts[0:2]]
    for point in sort_pts[2:]:
        while len(res_hull_top) >= 2 and not check_right(
                res_hull_top[-2], res_hull_top[-1], point):
            res_hull_top.pop()
        res_hull_top.append(point)

    rev_pts = np.flip(sort_pts, axis=0)
    res_hull_bottom = [*rev_pts[0:2]]
    for point in rev_pts[2:]:
        while len(res_hull_bottom) >= 2 and not check_right(
                res_hull_bottom[-2], res_hull_bottom[-1], point):
            res_hull_bottom.pop()
        res_hull_bottom.append(point)
    res_hull = np.asarray(res_hull_top + res_hull_bottom)
    return res_hull


class TestGramScan(unittest.TestCase):

    def test_check_right(self):
        p1 = np.asarray([0, 0])
        p2 = np.asarray([0, 1])
        p3 = np.asarray([1, 1])
        self.assertTrue(check_right(p1, p2, p3))

        p1 = np.asarray([0, 0])
        p2 = np.asarray([1, 0])
        p3 = np.asarray([1, 1])
        self.assertFalse(check_right(p1, p2, p3))

        p1 = np.asarray([0.1, 0.2])
        p2 = np.asarray([1, 0])
        p3 = np.asarray([1, 1])
        self.assertFalse(check_right(p1, p2, p3))


if __name__ == "__main__":
    # unittest.main()
    points = np.random.rand(10000, 2)
    num_points = np.logspace(2, 4, num=300)
    times = []
    for val in num_points:
        points = np.random.rand(int(val), 2)
        t0 = time.time()
        convex_hull = graham_scan(points)
        times.append(time.time() - t0)
    plt.plot(num_points, times)
    plt.plot(num_points, np.log(num_points) * num_points / 220000)
    plt.xscale('log')
    plt.show()
    # Test overlapping x values
    # points = np.concatenate([points, np.asarray([[points[0, 0], 0.5]])])

    # convex_hull = graham_scan(points)

    # plt.plot(convex_hull[:, 0], convex_hull[:, 1])
    # plt.scatter(points[:, 0], points[:, 1])
    # plt.show()
