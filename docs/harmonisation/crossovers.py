"""Where the lines of different datasets meet (plan.md, step 2.1): crossings, coincident tracks, and
clusters of both. Writes tables only, nothing into the database.

- **Segments:** each line cut at gaps of more than 2 km (turns are not cut yet, plan.md T9).
- **Crossings:** two segments of different datasets intersecting, on the lines thinned to 100 m. The
  depths are those of the nearest point of each line (its distance in sep_a, sep_b).
- **Coincident tracks:** runs of at least 1 km where a line of A stays within 100 m of lines of B (one
  or several: a flight of A can follow several short profiles of B), bridging stretches of up to 300 m
  where it moves up to 250 m away (Chung's flights weave 6 to 100 m around Mulvaney's).
  Sampled every km as pairs of kind "parallel". Crossings inside such a run are left out (the run holds
  them). Close tracks are not the same data (Chung and Mulvaney are different datasets): they are
  only places to compare, like crossings.
- **Clusters:** the pairs of points of one pair of lines within 2 km of each other (chained), and each
  coincident track, so that two lines meeting again and again, or one shared flight, count once.

    python docs/harmonisation/crossovers.py [database dir] [-o output dir]

Output (default ~/data/isochrones/harmonisation/<build>): segments.csv, runs.csv, pairs.parquet (one
row per pair of points: kind, dataset, file, index of each side, x, y, sep, cluster, run),
pair_depths.parquet (pair, side, IRH, age, depth), summary.md.
"""
import argparse
import collections
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from load import DEFAULT, load_lines  # noqa: E402

GAP = 2000.0  # m, cuts a line into segments
STEP = 100.0  # m, thinning for the search
PIECE = 50  # vertices per piece in the search tree (about 5 km)
CLOSE = 100.0  # m, coincident tracks
BRIDGE = 3  # thinned points (about 300 m) further than CLOSE, but within 2.5 CLOSE, kept inside a run
RUN_MIN = 1000.0  # m, shortest coincident track
SAMPLE = 1000.0  # m, spacing of the pairs along a coincident track
CLUSTER = 2000.0  # m


def thin(x, y, step=STEP):
    """Indices of the points kept, about step apart."""
    keep = [0]
    for i in range(1, len(x)):
        if np.hypot(x[i] - x[keep[-1]], y[i] - y[keep[-1]]) >= step:
            keep.append(i)
    if keep[-1] != len(x) - 1:
        keep.append(len(x) - 1)
    return np.array(keep)


def segments_of(lines):
    """(line, indices of the thinned points) per segment."""
    out = []
    for li, L in enumerate(lines):
        kept = L["kept"] = thin(L["x"], L["y"])
        jumps = np.flatnonzero(np.hypot(np.diff(L["x"][kept]), np.diff(L["y"][kept])) > GAP) + 1
        for part in np.split(kept, jumps):
            if len(part) >= 2:
                out.append((li, part))
    return out


def crossings(lines, segments):
    """Intersections of the segments of different datasets: (line a, line b, x, y)."""
    geoms, owner = [], []
    for li, idx in segments:
        for start in range(0, len(idx) - 1, PIECE):
            part = idx[start:start + PIECE + 1]
            geoms.append(shapely.linestrings(lines[li]["x"][part], lines[li]["y"][part]))
            owner.append(li)
    geoms, owner = np.array(geoms), np.array(owner)
    dataset = np.array([lines[li]["dataset"] for li in owner])
    a, b = shapely.STRtree(geoms).query(geoms, predicate="intersects")
    keep = (a < b) & (dataset[a] != dataset[b])
    a, b = a[keep], b[keep]
    points = shapely.intersection(geoms[a], geoms[b])
    out = []
    for i, j, p in zip(a, b, points):
        for q in shapely.get_parts(p):
            if shapely.get_type_id(q) == 0:  # points only: overlaps are coincident tracks
                out.append((owner[i], owner[j], q.x, q.y))
    return out


def coincident(lines, by_dataset):
    """Runs where a line of A (the first dataset in name order) stays within CLOSE of lines of B:
    (line a, thinned indices of a, line b and index of b at each, distances)."""
    runs = []
    names = sorted(by_dataset)
    thinned = {li: L["kept"] for li, L in enumerate(lines)}
    for ia, A in enumerate(names):
        for B in names[ia + 1:]:
            lb = by_dataset[B]
            pts_b = np.concatenate([np.c_[lines[li]["x"], lines[li]["y"]] for li in lb])
            owner_b = np.concatenate([np.full(len(lines[li]["x"]), li) for li in lb])
            index_b = np.concatenate([np.arange(len(lines[li]["x"])) for li in lb])
            tree = None
            for la in by_dataset[A]:
                L = lines[la]
                k = thinned[la]
                if tree is None:
                    tree = cKDTree(pts_b)
                dist, near = tree.query(np.c_[L["x"][k], L["y"][k]], distance_upper_bound=2.5 * CLOSE)
                hit = dist < CLOSE
                # Bridge short stretches a little further away
                miss = np.flatnonzero(~hit)
                if len(miss):
                    breaks = np.flatnonzero(np.diff(miss) > 1)
                    for a, b in zip(np.r_[miss[0], miss[breaks + 1]], np.r_[miss[breaks], miss[-1]]):
                        if a > 0 and b < len(k) - 1 and b - a < BRIDGE and np.isfinite(dist[a:b + 1]).all():
                            hit[a:b + 1] = True
                if not hit.any():
                    continue
                line_b = np.where(hit, owner_b[np.minimum(near, len(owner_b) - 1)], -1)
                # Consecutive thinned points near the same line of B
                start = None
                for p in range(len(k) + 1):
                    same = p < len(k) and hit[p]
                    if same and start is None:
                        start = p
                    elif not same and start is not None:
                        run = np.arange(start, p)
                        length = np.sum(np.hypot(np.diff(L["x"][k[run]]), np.diff(L["y"][k[run]])))
                        if length >= RUN_MIN:
                            runs.append((la, k[run], line_b[run], index_b[near[run]], dist[run]))
                        start = None
    return runs


def clusters(pairs):
    """Cluster id of each pair: a coincident track, or the crossings of one pair of lines chained within CLUSTER."""
    out = np.full(len(pairs), -1)
    next_id = 0
    runs = pairs[pairs["run"] >= 0]
    for _, members in runs.groupby("run"):
        out[members.index.to_numpy()] = next_id
        next_id += 1
    for _, group in pairs[pairs["run"] < 0].groupby(["line_a", "line_b"]):
        idx = group.index.to_numpy()
        parent = np.arange(len(idx))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in cKDTree(group[["x", "y"]].to_numpy()).query_pairs(CLUSTER):
            parent[root(i)] = root(j)
        roots = np.array([root(i) for i in range(len(idx))])
        _, ids = np.unique(roots, return_inverse=True)
        out[idx] = ids + next_id
        next_id += ids.max() + 1
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("database", nargs="?", default=DEFAULT)
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    out = Path(args.output or Path.home() / "data" / "isochrones" / "harmonisation" / Path(args.database).name)
    out.mkdir(parents=True, exist_ok=True)

    lines = load_lines(args.database)
    print(f"{len(lines)} lines")
    by_dataset = collections.defaultdict(list)
    for li, L in enumerate(lines):
        by_dataset[L["dataset"]].append(li)
    trees = {}

    def nearest(li, x, y):
        if li not in trees:
            trees[li] = cKDTree(np.c_[lines[li]["x"], lines[li]["y"]])
        return trees[li].query([x, y])

    segments = segments_of(lines)
    pd.DataFrame([dict(segment=s, dataset=lines[li]["dataset"], file=lines[li]["file"], line_id=lines[li]["line_id"],
                       first=int(idx[0]), last=int(idx[-1]),
                       length_km=float(np.sum(np.hypot(np.diff(lines[li]["x"][idx]), np.diff(lines[li]["y"][idx]))) / 1000))
                  for s, (li, idx) in enumerate(segments)]).to_csv(out / "segments.csv", index=False)
    print(f"{len(segments)} segments")

    runs = coincident(lines, by_dataset)  # after segments_of, which thins the lines
    print(f"{len(runs)} coincident tracks")
    run_rows, rows = [], []
    covered = collections.defaultdict(list)  # (line a, dataset b) -> thinned points of a in runs
    for r, (la, ka, lb, kb, dist) in enumerate(runs):
        L = lines[la]
        along = np.concatenate([[0], np.cumsum(np.hypot(np.diff(L["x"][ka]), np.diff(L["y"][ka])))])
        run_rows.append(dict(run=r, dataset_a=L["dataset"], file_a=L["file"], first_a=int(ka[0]), last_a=int(ka[-1]),
                             dataset_b=lines[lb[0]]["dataset"], lines_b=len(np.unique(lb)), length_km=along[-1] / 1000,
                             median_sep_m=float(np.median(dist))))
        covered[(la, lines[lb[0]]["dataset"])].append(np.c_[L["x"][ka], L["y"][ka]])
        for p in np.unique(np.searchsorted(along, np.arange(0, along[-1] + 1, SAMPLE))):
            p = min(p, len(ka) - 1)
            rows.append(dict(kind="parallel", line_a=la, index_a=int(ka[p]), line_b=int(lb[p]), index_b=int(kb[p]),
                             x=L["x"][ka[p]], y=L["y"][ka[p]], sep_a=0.0, sep_b=float(dist[p]), run=r))
    pd.DataFrame(run_rows).to_csv(out / "runs.csv", index=False)

    cross = crossings(lines, segments)
    inside = 0
    covered_trees = {k: cKDTree(np.concatenate(v)) for k, v in covered.items()}
    for la, lb, x, y in cross:
        if lines[la]["dataset"] > lines[lb]["dataset"]:
            la, lb = lb, la
        tree = covered_trees.get((la, lines[lb]["dataset"]))
        if tree is not None and tree.query([x, y])[0] < 2 * STEP:
            inside += 1
            continue
        da, ia = nearest(la, x, y)
        db, ib = nearest(lb, x, y)
        rows.append(dict(kind="crossing", line_a=la, index_a=int(ia), line_b=lb, index_b=int(ib), x=x, y=y,
                         sep_a=float(da), sep_b=float(db), run=-1))
    print(f"{len(cross)} crossings, {inside} of them inside coincident tracks (left out)")

    pairs = pd.DataFrame(rows)
    for side in "ab":
        pairs[f"dataset_{side}"] = [lines[li]["dataset"] for li in pairs[f"line_{side}"]]
        pairs[f"file_{side}"] = [lines[li]["file"] for li in pairs[f"line_{side}"]]
    pairs["cluster"] = clusters(pairs)
    depth_rows = []
    for pid, p in pairs.iterrows():
        for side in "ab":
            L = lines[p[f"line_{side}"]]
            d = L["depth"][p[f"index_{side}"]]
            for k in np.flatnonzero(np.isfinite(d)):
                depth_rows.append((pid, side, int(L["irh"][k]), L["age"][k], L["unc"][k], float(d[k])))
    pairs.drop(columns=["line_a", "line_b"]).to_parquet(out / "pairs.parquet")
    pd.DataFrame(depth_rows, columns=["pair", "side", "IRH", "age", "age_unc", "depth"]).to_parquet(out / "pair_depths.parquet")

    # Summary per pair of datasets
    run_df = pd.DataFrame(run_rows)
    g = pairs.groupby(["dataset_a", "dataset_b"])
    summary = pd.DataFrame({
        "crossings": g.apply(lambda d: int((d["kind"] == "crossing").sum())),
        "parallel pairs": g.apply(lambda d: int((d["kind"] == "parallel").sum())),
        "clusters": g["cluster"].nunique(),
    })
    if len(run_df):
        summary["coincident km"] = run_df.groupby(["dataset_a", "dataset_b"])["length_km"].sum().round(0)
    summary = summary.fillna(0).sort_values("clusters", ascending=False)
    with open(out / "summary.md", "w") as f:
        f.write(f"# Crossovers of {args.database}\n\n{len(lines)} lines, {len(segments)} segments, "
                f"{(pairs['kind'] == 'crossing').sum()} crossings ({inside} more inside coincident tracks), "
                f"{len(runs)} coincident tracks ({run_df['length_km'].sum() if len(run_df) else 0:.0f} km), "
                f"{pairs['cluster'].nunique()} clusters, over {len(summary)} pairs of datasets.\n\n")
        f.write(summary.reset_index().to_markdown(index=False))
        f.write("\n")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
