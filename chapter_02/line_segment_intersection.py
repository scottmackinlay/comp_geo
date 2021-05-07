from typing import Optional, List, Union, Tuple
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from matplotlib import pyplot as plt
from matplotlib import collections as mc


def gen_rand_lines(num_lines=10, grid_points: Optional[int] = None) -> Tuple[List, List]:
    """

    Args:
        num_lines: Number of lines to make
        grid_points: number of discrete x and y values to use

    Returns:
        A tuple comprising:
            list of points, each is a list of two floats
            list of lines, each is a list of indices into the points list

    """
    points = np.random.rand(num_lines * 2, 2)
    if grid_points:
        points = (points * grid_points).astype(int) / grid_points
    res_points = [list(p) for p in points]
    res_lines = [list(l) for l in np.arange(0, num_lines * 2).reshape(num_lines, 2)]
    return res_points, res_lines


def plot_lines(points: List, lines: List, show=True):
    points = np.asarray(points)
    lines = np.asarray(lines)
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
    points = np.asarray(points)
    lines = np.asarray(lines)

    lines_flipped = np.fliplr(lines)
    p1_y = points[lines[:, 0]][:, 1]
    p2_y = points[lines[:, 1]][:, 1]
    cond = p2_y < p1_y
    lines_oriented = np.where(np.tile(cond, (2, 1)).T, lines, lines_flipped)

    lines_sorted = lines_oriented[
        np.lexsort((points[lines_oriented[:, 0]][:, 0], points[lines_oriented[:, 0]][:, 1]), axis=0)]
    lines_sorted = np.flipud(lines_sorted)

    points = [list(p) for p in points]
    lines = [list(l) for l in lines_sorted]
    return points, lines


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


def calc_segment_intersection(a1: Union[List[int], np.ndarray],
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
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = [-da[1], da[0]]  # maybe convert to numpy
    u = (np.dot(dap, dp) / np.dot(dap, db))
    ip = u * db + b1
    v = np.dot(da, ip - a1) / np.linalg.norm(da) ** 2
    if u < 0 or u > 1 or v < 0 or v > 1:
        return None
    return ip


def modify_status(status: List[int],
                  event_queue: List[Event],
                  index_a: int,
                  index_b: int,
                  points: List,
                  lines: List):
    """
    Determines if there is an intersection between status[index_a] and status[index_b]
    If there is, it inserts (in place) an event into event_queue with event_type INTER
    """

    l1 = lines[status[index_a]]
    l2 = lines[status[index_b]]
    l1p1, l1p2 = (points[l1][0], points[l1][1])
    l2p1, l2p2 = (points[l2][0], points[l2][1])
    int_p = calc_segment_intersection(l1p1, l1p2, l2p1, l2p2)
    if int_p:
        points.append(int_p)
        inter_event = Event(len(points) - 1, EventType.INTER, [status[index_a], status[index_b]])
        for i, e in enumerate(event_queue):
            if point_comp(points[e.point_idx], int_p):
                event_queue.insert(i - 1, inter_event)
                break


def handle_event(event: Event, event_queue: List[Event], status: List[int], points: List, lines: List):
    if event.e_type == EventType.STARTPT:
        # add endpoint to event queue, insert that into status, modify status
        start_point_idx = lines[event.parents[0]][0]
        end_point_idx = lines[event.parents[0]][1]
        start_point = points[start_point_idx]
        end_point = points[end_point_idx]
        new_event = Event(end_point_idx, EventType.ENDPT, [event.parents[0]])

        stat_idx = len(status)
        for i, _ in enumerate(status):
            if point_comp(points[lines[status[i]][0]], start_point):
                stat_idx = i - 1
                break
        status.insert(stat_idx, event.parents[0])

        event_idx = len(event_queue)
        for i, e in enumerate(event_queue):
            if point_comp(points[e.point_idx], end_point):
                event_idx = i - 1
                break
        event_queue.insert(event_idx, new_event)

        if stat_idx != 0:
            modify_status(status, event_queue, stat_idx - 1, stat_idx, points, lines)
        if stat_idx <= len(status) - 2:
            modify_status(status, event_queue, stat_idx, stat_idx + 1, points, lines)

    if event.e_type == EventType.ENDPT:
        # remove line from status, check status
        l_idx_rem = event.parents[0]
        # The line had better be in the status array, or we've already screwed something up.
        status_l_idx = status.index(l_idx_rem)
        del status[l_idx_rem]
        # Only check status if the line was not on an end of the status array
        if status_l_idx != 0 and status_l_idx != len(lines):
            modify_status(status, event_queue, status_l_idx - 1, status_l_idx, points, lines)
    if event.e_type == EventType.INTER:
        # swap some stuff, check status
        parents = event.parents

        par_1_stat_idx = status.index(parents[0])
        par_2_stat_idx = status.index(parents[1])

        assert abs(par_1_stat_idx - par_2_stat_idx) < 2, "Parents are not adjacent in status. Investigate."
        par_left_stat_idx, par_right_stat_idx = sorted([par_1_stat_idx, par_2_stat_idx])

        temp_event = status[par_left_stat_idx]
        status[par_left_stat_idx] = status[par_right_stat_idx]
        status[par_right_stat_idx] = temp_event

        # Left status element check
        if par_left_stat_idx != 0:
            modify_status(status, event_queue, par_left_stat_idx - 1, par_left_stat_idx, points, lines)
        if par_right_stat_idx <= len(status) - 2:
            modify_status(status, event_queue, par_right_stat_idx, par_right_stat_idx + 1, points, lines)


def calc_intersections(points: np.ndarray, lines: np.ndarray):
    #TODO: Run through the algorithm and make sure that this makes sense.
    #"this is gonna work on the first try" -Scott
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
