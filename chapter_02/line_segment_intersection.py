from typing import Optional
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import collections  as mc
import random

def gen_rand_lines(num_lines = 10, grid_points: Optional[int] =None):
    # produces a #N, 2, 2 numpy array corresponding to a lot of lines
    points = np.random.rand(num_lines * 2, 2)
    if grid_points:
        points = (points * grid_points).astype(int) / grid_points
    return points, np.arange(0, num_lines * 2).reshape(num_lines, 2)

def plot_lines(points, lines, show=True):
    #expects numpy array of shape #N, 2, 2
    mp_lines = [[(points[l[0]][0], points[l[0]][1]), (points[l[1]][0], points[l[1]][1])] for l in lines]

    lc = mc.LineCollection(mp_lines, linewidths=2)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    ax.scatter(points[lines[:,0]][:,0], points[lines[:,0]][:,1], color="red")
    ax.scatter(points[lines[:,1]][:,0], points[lines[:,1]][:,1], color="blue")
    for i, l in enumerate(lines):
        ax.text((points[l[0]][0]), (points[l[0]][1]), i)
    
    if show:
        plt.show()
    return fig, ax

def line_sort(points, lines):
    lines_flipped = np.fliplr(lines)
    p1_y = points[lines[:,0]][:,1]
    p2_y = points[lines[:,1]][:,1]
    cond = p2_y < p1_y
    lines_oriented = np.where(np.tile(cond,(2,1)).T, lines, lines_flipped)
    
    lines_sorted = lines_oriented[np.lexsort((points[lines_oriented[:,0]][:,0], points[lines_oriented[:,0]][:,1]), axis=0)]
    return(points, np.flipud(lines_sorted))

def point_comp(p1, p2):
    """Returns True if p1 is down/left of p2, where down (y value) is first
    compared, and if the y values are close, left-right is compared.
    """
    if np.isclose(p1[1], p2[1]):
        return p1[0] < p2[0]
    else:
        return p1[1] < p2[1]


def calc_intersections(points: np.ndarray, lines: np.ndarray):
    _, lines = line_sort(points, lines)
    events = [(l, p, False) for l, p in enumerate(lines[:,0])]
    events.reverse()
    status = []
    min_x = np.min(points[:,0])
    max_x = np.max(points[:,0])
    
    queue_idx = 0
    while True:
        # Get the next event and handle it
        line_idx, point_idx, has_seen_line = events.pop()
        point = points[point_idx]
        # If the event point corresponds to a line we've added the first point
        # of, add the second point of the line to our events
        if not has_seen_line:
            
            line = lines[line_idx]
            print(points[line[1]])
            ins_idx = 0
            print(point, events)
            # TODO (Max and Scott): Insert the bottom point of the line into
            # our event list. 
            for event in events:
                if point_comp(points[line[1]], points[event[1]]):
                    ins_idx = event[1]
            print(ins_idx)
        # If it's a second point in a line, we remove the line from status or
        # something.

        # If it's an intersection, return it.

            # events.insert()
        # print(ins_idx)


        fig, plt_ax = plot_lines(points, lines, show=False)
        sweeping_line = mc.LineCollection([[(min_x, point[1]), (max_x, point[1])]], linewidths=2, color='black',)
        plt_ax.add_collection(sweeping_line)
        plt.show(block=True)

        if len(events) == 0:
            break


if __name__ == "__main__":
    # calc_intersections(gen_rand_lines(10, grid_points=10))
    calc_intersections(*gen_rand_lines(3))
