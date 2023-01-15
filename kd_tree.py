from typing import List
from collections import namedtuple
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn')

class Point(namedtuple("Point", "x y")):
    def __repr__(self) -> str:
        return f'Point{tuple(self)!r}'


class Rectangle(namedtuple("Rectangle", "lower upper")):
    def __repr__(self) -> str:
        return f'Rectangle{tuple(self)!r}'

    def is_contains(self, p: Point) -> bool:
        return self.lower.x <= p.x <= self.upper.x and self.lower.y <= p.y <= self.upper.y


class Node(namedtuple("Node", "location left right")):
    """
    location: Point
    left: Node
    right: Node
    """

    def __repr__(self):
        return f'{tuple(self)!r}'


class KDTree:
    """k-d tree"""

    def __init__(self):
        self._root = None
        self._n = 0

    def insert(self, p: List[Point]):
        """insert a list of points"""
        def insert_rec(p: List[Point], depth):
            if p == []:
                return None
            dim = depth % 2
            sorted_p = sorted(p, key=lambda p: p[dim])
            median_idx = len(p) // 2
            return Node(sorted_p[median_idx], insert_rec(sorted_p[:median_idx], depth + 1), insert_rec(sorted_p[median_idx + 1:], depth + 1))
        self._root = insert_rec(p,0)


    def range(self, rectangle: Rectangle) -> List[Point]:
        """range query"""
        def range_rec(rectangle: Rectangle, n: Node, depth):
            pts_in = []
            if rectangle.is_contains(n.location):
                pts_in = [n.location]
            dim = depth % 2    
            if  n.left and n.location[dim] >= rectangle.lower[dim]:
                pts_in.extend(range_rec(rectangle, n.left, depth + 1))
            if  n.right and n.location[dim] <= rectangle.upper[dim]:
                pts_in.extend(range_rec(rectangle, n.right, depth + 1))
            return pts_in
        return range_rec(rectangle, self._root, 0)

    
    def nearest_neighbor(self, target: Point):
        best_point = None
        best_dist = float("inf")
        root = self._root
        stack = [root]
        def distance(p1: Point, p2: Point):
            return (p1.x-p2.x)**2 + (p1.y-p2.y)**2
        while len(stack) > 0:
            node = stack.pop()
            dist = distance(node.location, target)
            if dist < best_dist:
                best_dist = dist
                best_point = node.location
            if node.left is not None and distance(node.left.location, target) < best_dist:
                stack.append(node.left)
            if node.right is not None and distance(node.right.location, target) < best_dist:
                stack.append(node.right)
        print(best_point)

def range_test():
    points = [Point(7, 2), Point(5, 4), Point(9, 6), Point(4, 7), Point(8, 1), Point(2, 3)]
    kd = KDTree()
    kd.insert(points)
    result = kd.range(Rectangle(Point(0, 0), Point(6, 6)))
    assert sorted(result) == sorted([Point(2, 3), Point(5, 4)])


def performance_test():
    points = [Point(x, y) for x in range(1000) for y in range(1000)]

    lower = Point(500, 500)
    upper = Point(504, 504)
    rectangle = Rectangle(lower, upper)
    #  naive method
    start = int(round(time.time() * 1000))
    result1 = [p for p in points if rectangle.is_contains(p)]
    end = int(round(time.time() * 1000))
    print(f'Naive method: {end - start}ms')
    naive_method = end - start

    kd = KDTree()
    kd.insert(points)
    # k-d tree
    start = int(round(time.time() * 1000))
    result2 = kd.range(rectangle)
    end = int(round(time.time() * 1000))
    print(f'K-D tree: {end - start}ms')
    kd_tree = end - start

    # Visualize the time performance between two methods
    names = ['kd_tree', 'naive_method']
    values = [kd_tree, naive_method]

    plt.figure(figsize=(11, 5))

    plt.subplot(131)
    plt.bar(names, values)
    plt.subplot(132)
    plt.scatter(names, values)
    plt.subplot(133)
    plt.plot(names, values)
    plt.suptitle('Time performance between two methods')
    plt.show()  
    assert sorted(result1) == sorted(result2)

def nn_test():
    points = [Point(7, 2), Point(5, 4), Point(9, 6), Point(4, 7), Point(8, 1), Point(2, 3)]
    kd = KDTree()
    kd.insert(points)
    kd.nearest_neighbor(Point(2,1))

if __name__ == '__main__':
    range_test()
    performance_test()
    nn_test()