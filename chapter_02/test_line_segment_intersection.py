import unittest
from line_segment_intersection import *


class TestLSI(unittest.TestCase):
    def test_gen_rand_lines(self):
        points, lines = gen_rand_lines(3, 10)
        self.assertTrue(all(isinstance(p, list) for p in points))
        self.assertTrue(all(isinstance(l, list) for l in lines))
        self.assertTrue(len(points) == 6)
        self.assertTrue(len(lines) == 3)

    def test_line_sort(self):
        points = [
            [2, 9],
            [10, 3],
            [2, 2],
            [1, 0],
            [8, 5],
            [4, 1]
        ]

        lines = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]

        points, lines = line_sort(points, lines)
        points = np.asarray(points)
        lines = np.asarray(lines)
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
        points = [
            [2, 9],
            [10, 3],
            [2, 6],
            [1, 2],
            [8, 5],
            [4, 1]
        ]

        lines = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]

        event_queue = [Event(lines[i][0], EventType.STARTPT, [i]) for i in range(3)]
        points, lines = line_sort(points, lines)

        points[2][1] = 2
        points[3][1] = 0
        points, lines = line_sort(points, lines)
        event = event_queue.pop(0)
        status = []
        handle_event(event, event_queue, status, points, lines)
        # TODO: test that adding starpoints modifies the status appropriately
        self.assertTrue(event_queue[1].e_type == EventType.ENDPT)

    def test_handle_event_endpt(self):
        points = [
            [2, 9],
            [10, 8],
            [2, 6],
            [1, 2],
            [8, 5],
            [4, 1]
        ]

        lines = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]
        status = [0]
        e = Event(1, EventType.ENDPT, [0])
        handle_event(e, [], status, points, lines)
        self.assertTrue(len(status) == 0)

    def test_handle_event_interpt(self):
        points = [
            [2, 9],
            [10, 3],
            [2, 6],
            [1, 2],
            [8, 5],
            [4, 1],
            [7.71, 4.71]
        ]

        lines = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]

        # plot_lines(points, lines)
        status = [1, 0, 2]
        e = Event(6, EventType.INTER, [0, 2])
        handle_event(e, [], status, points, lines)
        self.assertTrue(status == [1, 2, 0])

        status = [1, 0, 2]
        e = Event(6, EventType.INTER, [2, 0])
        handle_event(e, [], status, points, lines)
        self.assertTrue(status == [1, 2, 0])

    def test_point_comp(self):
        sim_event_points = np.asarray([
            [2, 9],
            [8, 5],
            [2, 2],
        ])
        res = [False, False, True]
        for p, r in zip(sim_event_points, res):
            self.assertEqual(point_comp(p, [10, 3]), r)

    def test_modify_status(self):
        points = [
            [3, 9],
            [10, 3],
            [2, 6],
            [1, 2],
            [9, 5],
            [4, 1],
        ]

        lines = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]
        pass
        # plot_lines(points, lines)
        # status = [0, 1]
        #
        # e = Event(6, EventType.INTER, [0, 2])
        # modify_status(status, [], )

    def test_intersections(self):
        points = [
            [3, 9],
            [10, 3],
            [2, 6],
            [1, 2],
            [9, 5],
            [4, 1],
        ]

        lines = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]
        calc_intersections(points, lines)




if __name__ == "__main__":
    unittest.main()
