#!/usr/bin/env python3
"""Exact certificate for the six-rectangle Venn diagram in the note.

The program uses fractions.Fraction throughout and requires only Python's
standard library.  It checks that the six curves are rectangles, constructs
the complete planar arrangement, traverses its faces, and verifies that the
64 face labels are exactly the 64 binary words.
"""

from collections import Counter, defaultdict, deque
from fractions import Fraction
from functools import cmp_to_key

D = 10_000
# Rows are (D*cx, D*cy, D*p, D*q, D*lambda).
PARAMETERS = [
    ( 3196,  -488, 6447,     0,  5191),
    (-2273,  -556, 7159, -2186,  4359),
    ( 1950,  1685, 5900,  3951,  4616),
    (  -28, -2245, 3588, -4020,  6507),
    (-2151,  -798, 7228,  3065,  3227),
    ( -694,  2402, 1890, -4201, 11274),
]

Point = tuple[Fraction, Fraction]


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def scale(t: Fraction, a: Point) -> Point:
    return (t * a[0], t * a[1])


def cross(a: Point, b: Point) -> Fraction:
    return a[0] * b[1] - a[1] * b[0]


def dot(a: Point, b: Point) -> Fraction:
    return a[0] * b[0] + a[1] * b[1]


def orient(a: Point, b: Point, c: Point) -> Fraction:
    return cross(sub(b, a), sub(c, a))


def rectangles():
    result = []
    for row in PARAMETERS:
        cx, cy, p, q, lam = (Fraction(value, D) for value in row)
        c = (cx, cy)
        u = (p, q)
        v = (-lam * q, lam * p)
        assert dot(u, v) == 0
        assert cross(u, v) > 0
        vertices = [
            sub(sub(c, u), v),
            add(sub(c, v), u),
            add(add(c, u), v),
            add(sub(c, u), v),
        ]
        assert all(
            orient(vertices[i], vertices[(i + 1) % 4], vertices[(i + 2) % 4]) > 0
            for i in range(4)
        )
        result.append(vertices)
    return result


def half(direction: Point) -> int:
    x, y = direction
    return 0 if (y > 0 or (y == 0 and x > 0)) else 1


def compare_directions(a: Point, b: Point) -> int:
    ha, hb = half(a), half(b)
    if ha != hb:
        return -1 if ha < hb else 1
    determinant = cross(a, b)
    if determinant > 0:
        return -1
    if determinant < 0:
        return 1
    la, lb = dot(a, a), dot(b, b)
    return -1 if la < lb else (1 if la > lb else 0)


def verify() -> None:
    rects = rectangles()
    edge_points = {
        (i, j): [
            (Fraction(0), rects[i][j]),
            (Fraction(1), rects[i][(j + 1) % 4]),
        ]
        for i in range(6)
        for j in range(4)
    }

    crossing_records = defaultdict(list)
    pair_counts = [[0] * 6 for _ in range(6)]
    crossings = 0

    for i in range(6):
        for k in range(i + 1, 6):
            for j in range(4):
                a, b = rects[i][j], rects[i][(j + 1) % 4]
                r = sub(b, a)
                for ell in range(4):
                    c, d = rects[k][ell], rects[k][(ell + 1) % 4]
                    s = sub(d, c)
                    o1, o2 = orient(a, b, c), orient(a, b, d)
                    o3, o4 = orient(c, d, a), orient(c, d, b)
                    assert o1 and o2 and o3 and o4
                    proper = ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0))
                    if not proper:
                        continue
                    denominator = cross(r, s)
                    assert denominator
                    t = cross(sub(c, a), s) / denominator
                    u = cross(sub(c, a), r) / denominator
                    assert 0 < t < 1 and 0 < u < 1
                    point = add(a, scale(t, r))
                    assert point == add(c, scale(u, s))
                    edge_points[(i, j)].append((t, point))
                    edge_points[(k, ell)].append((u, point))
                    crossing_records[point].append((i, k, j, ell))
                    pair_counts[i][k] += 1
                    crossings += 1

    assert crossings == 62
    assert len(crossing_records) == 62
    assert all(len(records) == 1 for records in crossing_records.values())

    corners = [point for rect in rects for point in rect]
    assert len(set(corners)) == 24
    assert not (set(corners) & set(crossing_records))

    edge_label = {}
    adjacency = defaultdict(set)
    for (curve, _side), points in edge_points.items():
        points.sort(key=lambda item: item[0])
        assert len({t for t, _point in points}) == len(points)
        for (_, a), (_, b) in zip(points, points[1:]):
            key = frozenset((a, b))
            assert key not in edge_label
            edge_label[key] = curve
            adjacency[a].add(b)
            adjacency[b].add(a)

    nodes = set(adjacency)
    vertex_count = len(nodes)
    edge_count = len(edge_label)
    assert vertex_count == 86
    assert edge_count == 148
    assert Counter(len(adjacency[p]) for p in nodes) == Counter({4: 62, 2: 24})

    seen = set()
    stack = [next(iter(nodes))]
    while stack:
        point = stack.pop()
        if point in seen:
            continue
        seen.add(point)
        stack.extend(adjacency[point] - seen)
    assert len(seen) == vertex_count

    ordered = {
        point: sorted(
            adjacency[point],
            key=cmp_to_key(
                lambda a, b, p=point: compare_directions(sub(a, p), sub(b, p))
            ),
        )
        for point in nodes
    }

    face_of = {}
    faces = []
    for a in nodes:
        for b in adjacency[a]:
            if (a, b) in face_of:
                continue
            face_id = len(faces)
            cycle = []
            current = (a, b)
            while current not in face_of:
                face_of[current] = face_id
                u, v = current
                cycle.append(u)
                neighbors = ordered[v]
                reverse_index = neighbors.index(u)
                w = neighbors[(reverse_index - 1) % len(neighbors)]
                current = (v, w)
            assert current == (a, b)
            twice_area = sum(
                cross(cycle[i], cycle[(i + 1) % len(cycle)])
                for i in range(len(cycle))
            )
            faces.append((cycle, twice_area / 2))

    face_count = len(faces)
    assert face_count == 64
    assert vertex_count - edge_count + face_count == 2
    outer_faces = [i for i, (_cycle, area) in enumerate(faces) if area < 0]
    assert len(outer_faces) == 1
    outer_face = outer_faces[0]
    assert all(area > 0 for i, (_cycle, area) in enumerate(faces) if i != outer_face)

    dual = defaultdict(list)
    for key, curve in edge_label.items():
        a, b = tuple(key)
        left_ab = face_of[(a, b)]
        left_ba = face_of[(b, a)]
        assert left_ab != left_ba
        dual[left_ab].append((left_ba, curve))
        dual[left_ba].append((left_ab, curve))

    labels = {outer_face: 0}
    queue = deque([outer_face])
    while queue:
        face = queue.popleft()
        for neighbor, curve in dual[face]:
            expected = labels[face] ^ (1 << curve)
            if neighbor in labels:
                assert labels[neighbor] == expected
            else:
                labels[neighbor] = expected
                queue.append(neighbor)

    assert len(labels) == 64
    assert set(labels.values()) == set(range(64))

    bounded_areas = [area for i, (_cycle, area) in enumerate(faces) if i != outer_face]
    symmetric_counts = [
        [pair_counts[i][j] if i < j else pair_counts[j][i] if j < i else 0 for j in range(6)]
        for i in range(6)
    ]

    print("EXACT VERIFICATION PASSED")
    print(f"proper crossings: {crossings}")
    print(f"arrangement graph: V={vertex_count}, E={edge_count}, F={face_count}")
    print(f"distinct face labels: {len(set(labels.values()))}")
    print(f"minimum bounded-face area: {float(min(bounded_areas)):.15g}")
    print("pairwise crossing matrix:")
    for row in symmetric_counts:
        print(" ".join(f"{value:2d}" for value in row))


if __name__ == "__main__":
    verify()
