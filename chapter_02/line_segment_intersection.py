from typing import Optional
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import collections  as mc

def gen_rand_lines(num_lines = 10, grid_points: Optional[int] =None):
    points = np.random.rand(num_lines * 2, 2)
    if grid_points:
        points = (points * grid_points).astype(int) / grid_points
    return points, np.arange(0, num_lines * 2).reshape(num_lines, 2)

def plot_lines(lines:np.ndarray, color_points=False):
    
    mp_lines = [[(l[0], l[1]), (l[2], l[3])] for l in lines]

    lc = mc.LineCollection(mp_lines, linewidths=2)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    ax.scatter(lines[:,0], lines[:,1], color="red")
    ax.scatter(lines[:,2], lines[:,3], color="blue")
    plt.show()

def line_sort(points, lines):
    # lines_yup = [None]*len(lines)
    
    # for i, line in enumerate(lines):
    #     if np.allclose(line[1], line[3]):
    #         if line[0] > line[2]:
    #             lines_yup[i] = np.concatenate([line[2:], line[0:2]])
    #         else:
    #             lines_yup[i] = line
    #     else:
    #         if line[1] < line[3]:
    #             lines_yup[i] = np.concatenate([line[2:], line[0:2]])
    #         else:
    #             lines_yup[i] = line
    # lines_yup = np.stack(lines_yup, axis= 0)
    # return lines_yup
    lines_flipped = np.fliplr(lines)
    p1_y = points[lines[:,0]][:,1]
    p2_y = points[lines[:,1]][:,1]
    cond = p2_y > p1_y
    print(np.tile(cond, (2,1)).T)
    print(np.where(np.tile(cond,(2,1)).T, lines, lines_flipped))

def calc_intersections(lines: np.ndarray):
    lines_yup = line_sort(lines)
    
    lines_lexsort = lines_yup[np.lexsort((lines_yup[:, 0], lines_yup[:, 1]), axis=0)]
    plot_lines(lines_lexsort)
    print(lines_lexsort)


if __name__ == "__main__":
    # plot_lines(gen_rand_lines(10))
    # calc_intersections(gen_rand_lines(10, grid_points=10))
    line_sort(*gen_rand_lines(10))