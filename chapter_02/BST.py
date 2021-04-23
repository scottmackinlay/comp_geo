
'''This was our first pass at implementing a BST (for use as the data structure
of the "event" in our line intersection code)

This is a WIP
'''

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
    

if __name__ == "__main__":
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