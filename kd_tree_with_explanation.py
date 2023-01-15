from typing import List
from collections import namedtuple
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn')

class Point(namedtuple("Point", "x y")): # Point is a namedtuple whose typename is "Point" and field_names are x, y.
    def __repr__(self) -> str: #  __repr__() is one of the magic method that returns a printable representation, which returns the simple string representation of the passed object.
        return f'Point{tuple(self)!r}'
# The fuction of a Point is to store the location of a node in two dimensions.


class Rectangle(namedtuple("Rectangle", "lower upper")): # Rectangle is a namedtuple whose typename is "Rectangle" and field_names are lower, upper.
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
            # if the inserted list becomes empty, the recursive process on this branch will stop and return None
                
            # To determine whether split by the x-axis or y-axis. 
            # The possible value of the index of the dimension is 0 or 1.
            # The reason why 2 is here is that we only consider the points in 2 dimensions.
            dim = depth % 2
            # Sort the list based on x or y
            sorted_p = sorted(p, key=lambda p: p[dim])
            # To get the index of the median value.
            median_idx = len(p) // 2
            # Split the sorted list into two parts and start the recursive operations.
            return Node(sorted_p[median_idx], insert_rec(sorted_p[:median_idx], depth + 1), insert_rec(sorted_p[median_idx + 1:], depth + 1))
        self._root = insert_rec(p,0)
            

    def range(self, rectangle: Rectangle) -> List[Point]:
        """range query"""
        def range_rec(rectangle: Rectangle, n: Node, depth):
            
            pts_in = [] # This is a dynamic list, which refreshes in each recursive operation.
                
            # If the node is in the rectangle the dynamic list will record it.
            if rectangle.is_contains(n.location):
                pts_in = [n.location]
            # Similar pathway to determine dimension in insert()
            dim = depth % 2   
            # If the location of the node is on the upper/right of the lower bound of the rectangle, 
            # it will go to the left subtree. 
            if  n.left and n.location[dim] >= rectangle.lower[dim]:
                # list.extend() can help us connect lists
                pts_in.extend(range_rec(rectangle, n.left, depth + 1))
            # If the location of the node is on the lower/left of the upper bound of the rectangle, 
            # it will go to the right subtree. 
            if  n.right and n.location[dim] <= rectangle.upper[dim]:
                pts_in.extend(range_rec(rectangle, n.right, depth + 1))
            return pts_in
        # In this method, we check whether in a rectangle from the root of the tree
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
    kd.insert(points) # build the tree
    result = kd.range(Rectangle(Point(0, 0), Point(6, 6))) # find the points in the rectangle
    assert sorted(result) == sorted([Point(2, 3), Point(5, 4)])


def performance_test():
    points = [Point(x, y) for x in range(1000) for y in range(1000)]

    lower = Point(500, 500)
    upper = Point(504, 504)
    rectangle = Rectangle(lower, upper) # construct the rectangle by two points
    #  naive method
    start = int(round(time.time() * 1000)) # record the start time of naive method
    result1 = [p for p in points if rectangle.is_contains(p)]
    end = int(round(time.time() * 1000)) # record the end time of naive method
    print(f'Naive method: {end - start}ms')
    naive_method = end - start

    kd = KDTree()
    kd.insert(points) # build the tree
    # k-d tree
    start = int(round(time.time() * 1000)) # record the start time of naive method
    result2 = kd.range(rectangle)
    end = int(round(time.time() * 1000)) # record the end time of naive method
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

# main method to activate the tests above
if __name__ == '__main__':
    range_test()
    performance_test()
    nn_test()