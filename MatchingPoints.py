import numpy as np


class MatchingPoints:
    def __init__(self):
        self.counter = 0
        self.matching_points = {}

    def add(self, matches):
        self.matching_points[self.counter] = matches
        self.counter += 1

    def unique(self):
        intersection_points = list(self.matching_points[0].keys())
        keys = list(self.matching_points.keys())
        for key in keys[1:]:
            intersection_points = [intersection_points[i] for i in range(len(intersection_points)) if
                                   intersection_points[i] in self.matching_points[key]]

        for key in self.matching_points:
            self.matching_points[key] = {k: self.matching_points[key][k] for k in self.matching_points[key] if k in intersection_points}

        matching_points = {}
        for i in range(len(intersection_points)):
            for key in self.matching_points:
                if intersection_points[i] not in matching_points:
                    matching_points[intersection_points[i]] = [self.matching_points[key][intersection_points[i]]]
                else:
                    matching_points[intersection_points[i]].append(self.matching_points[key][intersection_points[i]])
        self.matching_points = matching_points

    def reset(self):
        self.counter = 0
        self.matching_points = {}
