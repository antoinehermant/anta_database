"""Lines of layer files without line IDs, whose points may come in any order of pieces.

Sorting by PSX or PSY breaks on any line that turns, and always going to the nearest point jumps
between lines where they cross. So points are only linked as the files give them:
1. pieces: points following each other in a file without a gap (PIECE_GAP), joined through the
   points that files share;
2. pieces tracing the same profile with other coordinates (OVERLAP) become one, sorted along the
   longest path of the largest;
3. pieces are joined end to end, closest ends first, up to max_jump and only if they head the
   same way (MAX_TURN), so two lines ending at a crossing stay apart. Layers may overlap there
   with other coordinates (each end lies on the other piece): the points are then sorted along
   the joined line. Further apart, up to TRANSECT_GAP, only if both pieces and the gap between
   them are in one straight line (TRANSECT_TURN).
"""

from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

PIECE_GAP = 2000.0  # m
OVERLAP_DISTANCE = 100.0  # m
OVERLAP_SHARE = 0.5  # of the points of the smaller piece
MAX_TURN = 60.0  # degrees
TRANSECT_GAP = 30000.0  # m
TRANSECT_TURN = 15.0  # degrees
HEADING_LENGTH = 1000.0  # m along a piece, to measure its heading at its ends
NO_HEADING_GAP = 1000.0  # m: the furthest to join a piece too short to have a heading
MAX_STARTS = 20  # pieces order_line starts its chains from


class _Groups:
    """Union-find."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.parent[self.find(i)] = self.find(j)


def _longest_path(graph, members: np.ndarray) -> np.ndarray:
    """Indices (in members) of the longest path of a piece, from a double sweep."""
    if len(members) == 1:
        return np.array([0])
    sub = graph[members][:, members]
    start = int(np.argmax(dijkstra(sub, directed=False, indices=0)))
    distance, previous = dijkstra(sub, directed=False, indices=start, return_predecessors=True)
    node = int(np.argmax(distance))
    path = [node]
    while previous[node] >= 0:
        node = int(previous[node])
        path.append(node)
    return np.array(path[::-1])


def _along(xy: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Position of points along a path (x, y): at its nearest vertex, plus the way along it."""
    if len(path) == 1:
        return np.zeros(len(xy))
    steps = np.diff(path, axis=0)
    lengths = np.hypot(*steps.T)
    position = np.r_[0.0, np.cumsum(lengths)]
    tangents = steps / np.maximum(lengths, 1e-9)[:, None]
    tangents = np.vstack([tangents, tangents[-1]])
    nearest = cKDTree(path).query(xy)[1]
    return position[nearest] + np.einsum("ij,ij->i", xy - path[nearest], tangents[nearest])


def _heading(xy: np.ndarray) -> Optional[np.ndarray]:
    """Unit vector leaving a piece at the end of xy, None when too short to tell."""
    steps = np.hypot(*np.diff(xy[::-1], axis=0).T)
    reached = np.flatnonzero(np.cumsum(steps) >= min(HEADING_LENGTH, steps.sum())) if len(steps) else []
    if not len(reached):
        return None
    vector = xy[-1] - xy[-2 - reached[0]]
    length = np.hypot(*vector)
    return vector / length if length > OVERLAP_DISTANCE else None


def _distance_to_line(point: np.ndarray, line: np.ndarray) -> float:
    """Distance from a point to a polyline (x, y): lines have gaps, so not only to its points."""
    if len(line) == 1:
        return float(np.hypot(*(point - line[0])))
    start, step = line[:-1], np.diff(line, axis=0)
    along = np.einsum("ij,ij->i", point - start, step) / np.maximum(np.einsum("ij,ij->i", step, step), 1e-9)
    nearest = start + np.clip(along, 0, 1)[:, None] * step
    return float(np.hypot(*(point - nearest).T).min())


def split_into_lines(xy: np.ndarray, sequences: list[np.ndarray], max_jump: float) -> list[np.ndarray]:
    """The lines in the points xy (n, 2) in m, each as the indices of its points in order.
    sequences: for each file, the indices of its points in the order of the file."""
    n = len(xy)
    max_jump = float(max_jump)
    gap = min(PIECE_GAP, max_jump)
    starts, ends = [], []
    for sequence in sequences:
        steps = np.hypot(*np.diff(xy[sequence], axis=0).T)
        near = (steps <= gap) & (steps > 0)
        starts.append(sequence[:-1][near])
        ends.append(sequence[1:][near])
    # Once each: a step in several files is not longer
    starts, ends = np.unique(np.c_[np.concatenate(starts), np.concatenate(ends)], axis=0).T
    weights = np.hypot(*(xy[starts] - xy[ends]).T)
    graph = coo_matrix((weights, (starts, ends)), shape=(n, n)).tocsr()
    labels = connected_components(graph, directed=False)[1]
    order = np.argsort(labels, kind="stable")
    members = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
    members.sort(key=len, reverse=True)

    # Pieces tracing the same profile: a share of the points near, not a crossing
    trees = [cKDTree(xy[m]) for m in members]
    low = [xy[m].min(axis=0) - OVERLAP_DISTANCE for m in members]
    high = [xy[m].max(axis=0) + OVERLAP_DISTANCE for m in members]
    same = _Groups(len(members))
    for small in range(len(members) - 1, 0, -1):
        points = xy[members[small]]
        for large in range(small):
            box = (points >= low[large]).all(axis=1) & (points <= high[large]).all(axis=1)
            if box.mean() < OVERLAP_SHARE:
                continue
            close = trees[large].query(points[box], distance_upper_bound=OVERLAP_DISTANCE)[0]
            if np.isfinite(close).sum() >= OVERLAP_SHARE * len(points):
                same.union(small, large)
                break

    # Each group in order along the longest path of its largest piece (the first of the group)
    grouped: dict[int, list[int]] = {}
    for i in range(len(members)):
        grouped.setdefault(same.find(i), []).append(i)
    pieces = []
    for group in grouped.values():
        first = members[group[0]]
        path = xy[first[_longest_path(graph, first)]]
        points = np.concatenate([members[i] for i in group])
        pieces.append(points[np.argsort(_along(xy[points], path), kind="stable")])

    # End to end, closest first: end 0 is the start of a piece, end 1 its end
    ends_xy = np.array([xy[p[e]] for p in pieces for e in (0, -1)])
    headings = [_heading(xy[p[::-1]] if e == 0 else xy[p]) for p in pieces for e in (0, 1)]
    candidates = sorted(
        (float(np.hypot(*(ends_xy[a] - ends_xy[b]))), a, b)
        for a, b in cKDTree(ends_xy).query_pairs(max(max_jump, TRANSECT_GAP))
        if a // 2 != b // 2
    )
    limit = np.cos(np.radians(MAX_TURN))
    straight = np.cos(np.radians(TRANSECT_TURN))
    joined = _Groups(len(pieces))
    links: dict[int, int] = {}
    overlaps = set()
    for distance, a, b in candidates:
        if a in links or b in links or joined.find(a // 2) == joined.find(b // 2):
            continue
        ha, hb = headings[a], headings[b]
        if distance > max_jump:
            if ha is None or hb is None:
                continue
            way = (ends_xy[b] - ends_xy[a]) / distance
            if ha @ -hb < straight or ha @ way < straight or hb @ -way < straight:
                continue
        elif ha is None or hb is None:
            if distance > NO_HEADING_GAP:
                continue
        else:
            way = (ends_xy[b] - ends_xy[a]) / max(distance, 1e-9)
            overlap = (
                _distance_to_line(ends_xy[a], xy[pieces[b // 2]]) <= OVERLAP_DISTANCE
                and _distance_to_line(ends_xy[b], xy[pieces[a // 2]]) <= OVERLAP_DISTANCE
            )
            if ha @ -hb < limit:
                continue
            if overlap:
                overlaps.update((a, b))
            elif distance > OVERLAP_DISTANCE and (ha @ way < limit or hb @ -way < limit):
                continue
        links[a], links[b] = b, a
        joined.union(a // 2, b // 2)

    lines, seen = [], set()
    for start in range(len(pieces)):
        for end in (0, 1):
            if start in seen or 2 * start + end in links:
                continue
            # This end is free: follow the links from it, or take a piece without links
            line, path, piece, entry, overlapping = [], [], start, end, False
            while True:
                seen.add(piece)
                ordered = pieces[piece] if entry == 0 else pieces[piece][::-1]
                line.append(ordered)
                # The path of the line leaves out the part of a piece overlapping the previous one
                skip = cKDTree(xy[ordered]).query(xy[path[-1][-1]])[1] if path and 2 * piece + entry in overlaps else 0
                overlapping = overlapping or skip > 0
                path.append(ordered[skip:])
                other = links.get(2 * piece + (1 - entry))
                if other is None:
                    break
                piece, entry = other // 2, other % 2
            points = np.concatenate(line)
            if overlapping:
                points = points[np.argsort(_along(xy[points], xy[np.concatenate(path)]), kind="stable")]
            lines.append(points)
    return sorted(lines, key=lambda line: int(line.min()))


def path_length(xy: np.ndarray) -> float:
    """Length (m) of the path through the points xy (n, 2) in m, in their order."""
    finite = np.isfinite(xy).all(axis=1)
    return float(np.hypot(*np.diff(xy[finite], axis=0).T).sum())


def order_line(xy: np.ndarray, sequences: list[np.ndarray], max_jump: float = PIECE_GAP) -> np.ndarray:
    """The points xy (n, 2) in m of one line, in order, as indices: the lines of split_into_lines
    joined end to end. Of the chains that start from each end of each piece and go to the closest
    free end, the one with the shortest jumps in total: a line with its own ID is never cut.
    sequences: for each file, the indices of its points in the order of the file."""
    pieces = split_into_lines(xy, sequences, max_jump)
    if len(pieces) == 1:
        return pieces[0]
    ends = np.array([[xy[p[0]], xy[p[-1]]] for p in pieces])  # (piece, start or end, x y)
    best, best_length = None, np.inf
    # Each start costs (number of pieces)^2: from the longest pieces only, when there are many
    starts = sorted(range(len(pieces)), key=lambda k: -len(pieces[k]))[:MAX_STARTS]
    for first in starts:
        for reverse in (False, True):
            chain, length = [(first, reverse)], 0.0
            left = set(range(len(pieces))) - {first}
            while left:
                k, rev = chain[-1]
                here = ends[k, 0 if rev else 1]
                # The closest end of a piece left: entered by its start, or by its end (reversed)
                k, rev, jump = min(
                    ((j, side == 1, float(np.hypot(*(ends[j, side] - here)))) for j in left for side in (0, 1)),
                    key=lambda c: c[2],
                )
                chain.append((k, rev))
                length += jump
                left.discard(k)
            if length < best_length:
                best, best_length = chain, length
    return np.concatenate([pieces[k][::-1] if rev else pieces[k] for k, rev in best])


def out_of_order(xy: np.ndarray) -> bool:
    """Whether the points xy (n, 2) in m of a line go back to where they were: more than 2 jumps
    (over 2 km) that land within 1 km of a point passed before. A gap in the picks is not."""
    xy = xy[np.isfinite(xy).all(axis=1)]
    steps = np.hypot(*np.diff(xy, axis=0).T)
    jumps = np.flatnonzero(steps > 2000)
    if not len(jumps):
        return False
    tree = cKDTree(xy)
    back = sum(any(k < j - 10 for k in tree.query_ball_point(xy[j + 1], 1000)) for j in jumps)
    return back > 2


def order_by_files(xy: np.ndarray, sequences: list[np.ndarray], max_jump: float = PIECE_GAP) -> Optional[np.ndarray]:
    """The points xy (n, 2) in m of one line in an order that keeps the order of the files, as
    indices; None if they disagree. For a flight passing again near its own track (out and back,
    loops at the turns), which order_line can weave together where it joins pieces by their
    closest ends: the files, in the order of the flight, keep the passes apart.
    sequences: for each file, the indices of its points in the order of the file. A file is in
    the order of the flight only between its gaps (over max_jump: a file may list the parts of
    a line in another order), so its pieces are turned to the way of the longest piece (by the
    points they share). The points are then taken one after the other: of those whose
    predecessors in every piece are taken, the closest."""
    n = len(xy)
    pieces = []
    for seq in sequences:
        seq = np.asarray(seq)
        cut = np.flatnonzero(np.hypot(*np.diff(xy[seq], axis=0).T) > max_jump) + 1
        pieces += [p for p in np.split(seq, cut) if len(p)]
    pieces.sort(key=len, reverse=True)
    # Where along the longest piece each point is, carried over to the pieces turned to its way
    rank = np.full(n, np.nan)
    oriented = []
    for seq in pieces:
        known = np.isfinite(rank[seq])
        shared = rank[seq][known]
        if known.sum() >= 2 and shared.std() > 0 and np.corrcoef(np.flatnonzero(known), shared)[0, 1] < 0:
            seq, known = seq[::-1], known[::-1]
        oriented.append(seq)
        steps = np.arange(len(seq), dtype=float)
        if known.sum() >= 2:
            rank[seq[~known]] = np.interp(steps[~known], steps[known], rank[seq][known])
        elif not oriented[:-1]:
            rank[seq] = steps
    successors = [[] for _ in range(n)]
    waiting = np.zeros(n, dtype=int)
    for seq in oriented:
        for a, b in zip(seq[:-1], seq[1:]):
            successors[a].append(b)
            waiting[b] += 1
    ready = set(np.flatnonzero(waiting == 0).tolist())
    order, here = [], xy[oriented[0][0]]
    while ready:
        candidates = np.fromiter(ready, dtype=int)
        k = int(candidates[np.argmin(np.nan_to_num(np.hypot(*(xy[candidates] - here).T), nan=np.inf))])
        ready.discard(k)
        order.append(k)
        here = xy[k]
        for b in successors[k]:
            waiting[b] -= 1
            if waiting[b] == 0:
                ready.add(b)
    # A point left out waits for itself through the files: they disagree on the order
    return np.array(order) if len(order) == n else None
