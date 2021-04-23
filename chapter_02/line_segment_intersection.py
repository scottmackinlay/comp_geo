from typing import Optional
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import collections  as mc
import random

class BST():
    def __init__(self, data, comp_func):
        self.comp_func = comp_func
        self.data = data
        self.left_child = None
        self.right_child = None
    

    def insert_point(self, data):
        is_greater = self.comp_func(data, self.data)
        if not is_greater:
            if self.right_child is None:
                self.right_child = BST(data, self.comp_func)
            else:
                self.right_child.insert_point(data)
        else:
            if self.left_child is None:
                self.left_child = BST(data, self.comp_func)
            else:
                self.left_child.insert_point(data)
    
    def depth(self):
        lc_depth = 1
        rc_depth = 1
        if self.right_child is not None:
            rc_depth =  1 + self.right_child.depth()      
        if self.left_child is not None:
            lc_depth = 1 + self.left_child.depth()

        return max(rc_depth, lc_depth)

    def __repr__(self):
        depth = self.depth()
        chars = 2**(depth+1)
        pad = " "*chars
        
        lc_str = "X"
        rc_str = "X"
        if self.left_child is not None:
            lc_str = self.left_child.__repr__()
        if self.right_child is not None:
            rc_str = self.right_child.__repr__()

        cur_line = f"{pad}{depth}{pad}"
        pipe_line = ""
        child_line = ""

        if self.left_child is None and self.right_child is None:
            return f"{self.data}" 
        return f"({self.data}, {lc_str}, {rc_str})"
        # return f"{pad}{depth}{pad}\n{lc_str}{pad[:int(chars/2)]}{rc_str}"
        

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

def calc_intersections(points: np.ndarray, lines: np.ndarray):
    _, lines_sorted = line_sort(points, lines)
    
    
    queue = lines_sorted[:,0]
    status = []
    min_x = np.min(points[:,0])
    max_x = np.max(points[:,0])
    
    queue_idx = 0
    while True:
        pt_idx = queue[queue_idx]
        pt = points[pt_idx]
        
        # Add point to queue
        # queue.insert
        
        status.append(queue_idx)
        
        
        
        
        
        fig, plt_ax = plot_lines(points, lines_sorted, show=False)

        sweeping_line = mc.LineCollection([[(min_x, pt[1]), (max_x, pt[1])]], linewidths=2)

        plt_ax.add_collection(sweeping_line)
        plt.show(block=True)
        queue_idx += 1
        if queue_idx == len(queue):
            break





if __name__ == "__main__":
    # calc_intersections(gen_rand_lines(10, grid_points=10))
    # calc_intersections(*gen_rand_lines(3))
    def comp_func(p1, p2): 
        return p1 < p2
    
    p0 = 5
    bst = BST(p0, comp_func)
    # print(bst)
    for i in range(6):
        insert = random.randint(0,10)
        bst.insert_point(insert)
        print(insert)
    print(bst)
    print(bst.depth())