#!/usr/bin/env python
"""
Showcases the answer of a stackoverflow question:
https://stackoverflow.com/a/1407395/177931

Implements an analytic solution based on what is documented in above answer.
"""

import numpy as np


def get_spanning_vectors_of_3d_plane(n):
    """
    Gets two orthogonal spanning vectors of a plane with normal n.
    :param n: (array_like) normal of plane of dimension 3, does not have to be normalized, but is normalized internally.
    :return: u,v (ndarray) of dimension 3 which are the spanning vectors.

    Note this function uses random vectors to initialize guesses which may be undesirable,
    this can easily be changed to use deterministic guesses but requires more checks.
    """

    n = np.asarray(n)
    n = n / np.linalg.norm(n)

    u = np.array([-n[1], n[0], 0])
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)

    return u, v


def eval_circle(t, circle):
    """
    Evaluates a circle in 3d.
    :param t: (ndarray, domain=[0:2*pi)) a ndarray of one or more angles at which the 3d circle will be evaluated.
    :param circle: (circle) the circle in 3d to evaluate.
    :return: (ndarray) a ndarray of the 3d points that each value of t maps to, on on each row.

    Note for a description of the circle structure, please see documentation of Sphere.get_circle_of_intersection.
    """
    h, c, u, v = circle
    circle_pts = c + h * np.cos(t) * u + h * np.sin(t) * v
    return circle_pts


def solve_sin_cos_eq(alpha, beta, gamma):
    """
    Solves the equation
    \alpha cos(t) + \beta sin(t) = \gamma
    for t.

    :param alpha: (real)
    :param beta: (real)
    :param gamma: (real)
    :return: all solutions for t to the equation.
    """

    # Solve for tt = tan(t/2) in the original equations using
    # Weierstrass Substitution:
    # cos(t) = (1-tt^2)/(1+tt^2)
    # sin(t) = (2tt)/(1+tt^2)

    c = gamma - alpha
    b = -2 * beta
    a = gamma + alpha
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return np.nan, np.nan

    denom = 1 / (2 * a)
    discrim_root = np.sqrt(discriminant)
    tt_0 = (-b + discrim_root) * denom
    tt_1 = (-b - discrim_root) * denom

    t_0 = 2 * np.arctan(tt_0)
    t_1 = 2 * np.arctan(tt_1)

    t_0 = t_0 if t_0 > 0 else t_0 + 2 * np.pi
    t_1 = t_1 if t_1 > 0 else t_1 + 2 * np.pi

    return t_0, t_1


class Sphere:

    def __init__(self, center_sigma=1., radius_sigma=2.):
        """
        Generates a random sphere in 3d.
        :param center_sigma: std. deviation in random center.
        :param radius_sigma: std. deviation in random radius.
        """
        self.center = center_sigma * np.random.randn(3)
        self.radius = radius_sigma * np.abs(np.random.randn())

    def check_intersection(self, other_sphere):
        """
        Checks to see if this sphere intersects with given `other_sphere`.
        :param other_sphere:
        :return:
        """
        r0 = other_sphere.radius
        c0 = other_sphere.center
        r1 = self.radius
        c1 = self.center
        if r1 > r0:
            r0, r1 = r1, r0
            c0, c1 = c1, c0
        d = np.linalg.norm(c1 - c0)
        if d < r0:
            return r1 >= r0 - d
        else:
            return d < r0 + r1

    def get_circle_of_intersection(self, other_sphere):
        """
        Gets the circle in 3d of the intersection of this sphere with `other_sphere`.
        Note, an intersection should exist for this function to be sensible.
        :param other_sphere:
        :return: (circle) the circle of intersection.

        Note: a `circle` is defined by a tuple
        circle = h, c, u, v
        which describes the analytic form
        x(t) = c + h*cos(t)*u + h*sin(t)*v
        where:
        h is a constant (radius),
        c, u, v are all 3d vectors, and
        x(t) is the 3d circle as a function of time.
        """
        r0 = other_sphere.radius
        c0 = other_sphere.center
        r1 = self.radius
        c1 = self.center
        if r1 > r0:
            r0, r1 = r1, r0
            c0, c1 = c1, c0
        n = c1 - c0
        d = np.linalg.norm(n)
        n_normed = n / d

        u, v = get_spanning_vectors_of_3d_plane(n_normed)

        x = (r1 * r1 - r0 * r0 + d * d) / (2 * d)
        c = c0 + n_normed * (d - x)
        h = np.sqrt(r1 * r1 - x * x)
        return h, c, u, v

    def check_point_is_on_sphere(self, x):
        """
        Checks to see if a point is on a sphere (in 3d.)
        :param x:  (array_like) a 3d real point.
        :return: true if detected on sphere (within precision) false otherwise.
        """
        dx = x - self.center
        return np.abs(np.sum(np.square(dx)) - np.square(self.radius)) < 1e-6

    def find_intersection_with_circle(self, circle):
        """
        Finds the two points that define the intersection to this sphere with `circle`, if such an intersection exists.
        :param circle: (circle) the circle to intersect with.
        :return: (tuple) two 3d points that define the intersection. If no intersection exists returns None.
        """
        h, c, u, v = circle
        cd = c - self.center
        gamma = np.square(self.radius) - np.dot(cd, cd) - np.square(h)
        alpha = 2 * np.dot(cd, u) * h
        beta = 2 * np.dot(cd, v) * h

        t0, t1 = solve_sin_cos_eq(alpha, beta, gamma)

        if np.isnan(t0) or np.isnan(t1):
            return None

        p0 = eval_circle(t0, circle)[0]
        p1 = eval_circle(t1, circle)[0]
        return p0, p1


# Set a random seed that creates an example where all three spheres intersect (trial-and-error finds one)
np.random.seed(1012)
# np.random.seed(1088) # another interesting example
# Create the three 3d spheres to check for intersection
print('Generating 3 random spheres.')
spheres = [Sphere() for _ in range(3)]
# If the first two spheres do not intersect, no point continuing, explain this to the user and quit early.
if not spheres[0].check_intersection(spheres[1]):
    raise RuntimeError(
        "In the generated example the first two spheres do not intersect; no 3-sphere intersection is possible.")
# Get the circle of intersection of first and second sphere
circle = spheres[0].get_circle_of_intersection(spheres[1])
# Get the two points of intersection of the circle with the third sphere
ret_value = spheres[2].find_intersection_with_circle(circle)
if ret_value:
    p0, p1 = ret_value
    p0_on_all_spheres = all([s.check_point_is_on_sphere(p0) for s in spheres])
    p1_on_all_spheres = all([s.check_point_is_on_sphere(p1) for s in spheres])
    print('The two 3d points where all 3 spheres intersect are:')
    print('P0 = ', p0)
    print('P1 = ', p1)
    print(f'P0 exists on all three spheres: {p0_on_all_spheres}')
    print(f'P1 exists on all three spheres: {p1_on_all_spheres}')
else:
    print('No intersection exists for the circle and the 3rd sphere.')
