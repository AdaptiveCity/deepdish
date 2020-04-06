import numpy as np
import sys

def intersection(p,pr,q,qs):
    # stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    r = pr - p
    s = qs - q
    rxs = np.cross(r,s)
    qmp = q - p
    qpxr = np.cross(qmp, r)
    if abs(rxs) < sys.float_info.epsilon:
        if abs(qpxr) < sys.float_info.epsilon:
            # co-linear
            rdrr = r / np.dot(r, r)
            t0 = np.dot(qmp, rdrr)
            t1 = t0 + np.dot(s, rdrr)
            if t0 > t1:
                t0, t1 = t1, t0
            return not (t1 < 0 or t0 > 1)
        else:
            return False # parallel, non-intersecting
    t = np.cross(qmp, s) / rxs
    u = qpxr / rxs
    return 0.0 <= t and t <= 1.0 and 0.0 <= u and u <= 1.0

def any_intersection(p1,q1,pts):
    for p2, q2 in zip(pts,pts[1:]):
        if intersection(p1,q1,p2,q2):
            return True
    return False

def f(x):
    return np.array(x,dtype=float)

p1 = f([0,0])
q1 = f([1,0])
p2 = f([1,-1])
q2 = f([0,1])
assert intersection(p1,q1,p2,q2) == True

p3 = f([1,2])
q3 = f([1,1])
assert intersection(p1,q1,p3,q3) == False

p4 = f([1.01,0])
q4 = f([2,0])
assert intersection(p1,q1,p4,q4) == False

p5 = f([1,2])
q5 = f([1,3])
assert intersection(p3,q3,p5,q5) == True

pts1 = f([[1,2],[1,1],[1,-1],[1,-2]])
assert any_intersection(p1,q1,pts1) == True

pts2 = f([[1,2],[1,1],[3,1],[3,-2]])
assert any_intersection(p1,q1,pts2) == False

