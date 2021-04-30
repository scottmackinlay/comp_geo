from typing import Optional, List, Union, Tuple
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from matplotlib import pyplot as plt
from matplotlib import collections  as mc


def gen_rand_lines(num_lines=10, grid_points: Optional[int] = None):
    # produces a #N, 2, 2 numpy array corresponding to a lot of lines
    points = np.random.rand(num_lines * 2, 2)
    if grid_points:
        points = (points * grid_points).astype(int) / grid_points
    return points, np.arange(0, num_lines * 2).reshape(num_lines, 2)


def plot_lines(points: List[np.ndarray], lines: np.ndarray, show=True):
    # expects numpy array of shape #N, 2, 2
    mp_lines = [[(points[l[0]][0], points[l[0]][1]), (points[l[1]][0], points[l[1]][1])] for l in lines]

    lc = mc.LineCollection(mp_lines, linewidths=2)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    ax.scatter(points[lines[:, 0]][:, 0], points[lines[:, 0]][:, 1], color="red")
    ax.scatter(points[lines[:, 1]][:, 0], points[lines[:, 1]][:, 1], color="blue")
    for i, l in enumerate(lines):
        ax.text(np.mean(points[l][:, 0]), np.mean(points[l][:, 1]), i)

    if show:
        plt.show()
    return fig, ax


def line_sort(points, lines):
    lines_flipped = np.fliplr(lines)
    p1_y = points[lines[:, 0]][:, 1]
    p2_y = points[lines[:, 1]][:, 1]
    cond = p2_y < p1_y
    lines_oriented = np.where(np.tile(cond, (2, 1)).T, lines, lines_flipped)

    lines_sorted = lines_oriented[
        np.lexsort((points[lines_oriented[:, 0]][:, 0], points[lines_oriented[:, 0]][:, 1]), axis=0)]
    lines_sorted = np.flipud(lines_sorted)

    return points, lines_sorted



def point_comp(p1, p2):
    """Returns True if p1 is down/left of p2, where down (y value) is first
    compared, and if the y values are close, left-right is compared.
    """
    if np.isclose(p1[1], p2[1]):
        return p1[0] < p2[0]
    else:
        return p1[1] < p2[1]


class EventType(IntEnum):
    STARTPT = 0
    ENDPT = 1
    INTER = 2


@dataclass
class Event:
    point_idx: int
    e_type: EventType
    parents: List[int]


def calc_segment_intersection(a1:  Union[List[int], np.ndarray],
                              a2: Union[List[int], np.ndarray],
                              b1: Union[List[int], np.ndarray],
                              b2: Union[List[int], np.ndarray]) -> Optional[np.ndarray]:
    """
    Calculates the intersections between lines a1->a2 and b1->b2. If no intersection is found, None is returned.
    """
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    b1 = np.asarray(b1)
    b2 = np.asarray(b2)
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = [-da[1], da[0]] # maybe convert to numpy
    u = (np.dot(dap, dp) / np.dot(dap, db))
    ip = u * db + b1
    v = np.dot(da, ip - a1)/np.linalg.norm(da)**2
    if u < 0 or u > 1 or v < 0 or v > 1:
        return None
    return ip


def check_status(status: List[int],
                 index_a: int,
                 index_b: int,
                 points: np.ndarray,
                 lines: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Event]]:
    l1 = lines[status[index_a]]
    l2 = lines[status[index_b]]
    l1p1, l1p2 = points[l1]
    l2p1, l2p2 = points[l2]
    l_int = calc_segment_intersection(l1p1, l1p2, l2p1, l2p2)
    # if l_int:
    #     return Event(len(points), )

def handle_event(event: Event, event_queue: List[Event], status: List[int], points: np.ndarray, lines: np.ndarray):
    # TODO: Turn point array into a list of points so we can append in place inside check_status
    # TODO: Bonus points if you make lines a list as well for consistency

    if event.e_type == EventType.STARTPT:
        # add endpoint to event queue, insert that into status, check status
        end_point_idx = lines[event.parents[0]][1]

        end_point = points[end_point_idx]
        new_event = Event(end_point_idx, EventType.ENDPT, [event.parents[0]])
        for i, e in enumerate(event_queue):
            if point_comp(points[e.point_idx], end_point):
                event_queue.insert(i-1, new_event)
                return
        event_queue.insert(len(event_queue), new_event)
    if event.e_type == EventType.ENDPT:
        # remove line from status, check status
        pass
    if event.e_type == EventType.INTER:
        # swap some stuff, check status
        pass

def calc_intersections(points: np.ndarray, lines: np.ndarray):
    _, lines = line_sort(points, lines)
    events = [(l, p, False) for l, p in enumerate(lines[:, 0])]
    events.reverse()
    status = []
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])

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
        sweeping_line = mc.LineCollection([[(min_x, point[1]), (max_x, point[1])]], linewidths=2, color='black', )
        plt_ax.add_collection(sweeping_line)
        plt.show(block=True)

        if len(events) == 0:
            break


if __name__ == "__main__":
    # calc_intersections(gen_rand_lines(10, grid_points=10))
    # calc_intersections(*gen_rand_lines(3))

    pass


