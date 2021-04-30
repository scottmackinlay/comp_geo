import unittest
from line_segment_intersection import *


class TestLSI(unittest.TestCase):
    def test_line_sort(self):
        points = np.asarray([
            [2, 9],
            [10, 3],
            [2, 2],
            [1, 0],
            [8, 5],
            [4, 1]
        ])

        lines = np.asarray([
            [0, 1],
            [2, 3],
            [4, 5]
        ])

        points, lines = line_sort(points, lines)
        self.assertTrue(np.all(np.diff(points[lines[:, 1]][:, 1]) < 0))

    def test_calculate_segment_intersection(self):
        res = calc_segment_intersection([1, 1], [7, 7], [1, 7], [7, 1])
        self.assertTrue(np.allclose(res, np.asarray([4, 4])))

        res = calc_segment_intersection([1, 1], [2, 2], [1, 7], [7, 1])
        self.assertEqual(res, None)

        res = calc_segment_intersection([1, 1], [7, 7], [1, 7], [2, 6])
        self.assertEqual(res, None)

        res = calc_segment_intersection([1, 1], [7, 7], [6, 2], [7, 1])
        self.assertEqual(res, None)

        res = calc_segment_intersection([6, 6], [7, 7], [1, 7], [7, 1])
        self.assertEqual(res, None)

    def test_handle_event_startpt(self):
        points = np.asarray([
            [2, 9],
            [10, 3],
            [2, 6],
            [1, 2],
            [8, 5],
            [4, 1]
        ])

        lines = np.asarray([
            [0, 1],
            [2, 3],
            [4, 5]
        ])

        event_queue = [Event(lines[i][0], EventType.STARTPT, [i]) for i in range(3)]
        points, lines = line_sort(points, lines)

        points[2, 1] = 2
        points[3, 1] = 0
        points, lines = line_sort(points, lines)
        event = event_queue.pop(0)

        handle_event(event, event_queue, [], points, lines)
        self.assertTrue(event_queue[1].e_type == EventType.ENDPT)

    def test_point_comp(self):
        sim_event_points = np.asarray([
            [2, 9],
            [8, 5],
            [2, 2],
        ])
        res = [False, False, True]
        for p, r in zip(sim_event_points, res):
            self.assertEqual(point_comp(p, [10, 3]), r)


if __name__ == "__main__":
    unittest.main()
