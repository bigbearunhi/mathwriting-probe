"""Streaming MathWriting -> disk-backed cubic-curve features.

Independent approximation of Carbune et al.; numerical choices are explicit.
No ground-truth-dependent resampling, and no train/validation/test mixing.
"""
import argparse
from collections import Counter
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tarfile
import time
import xml.etree.ElementTree as ET

import numpy as np
from probe import tokenize

NS = '{http://www.w3.org/2003/InkML}'


def minimum_frames(ids):
    return len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))


def fit_curve(points):
    """Endpoint-constrained cubic with chord-length parametrization."""
    if len(points) == 1:
        return np.repeat(points, 4, axis=0), np.zeros(1)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(distances)]
    u = cumulative/cumulative[-1] if cumulative[-1] > 1e-10 else np.linspace(0, 1, len(points))
    p0, p3 = points[0], points[-1]
    b0, b1, b2, b3 = (1-u)**3, 3*u*(1-u)**2, 3*u*u*(1-u), u**3
    if len(points) < 4:
        c1, c2 = p0+(p3-p0)/3, p0+2*(p3-p0)/3
    else:
        c1, c2 = np.linalg.lstsq(np.stack([b1, b2], 1),
                 points-b0[:, None]*p0-b3[:, None]*p3, rcond=None)[0]
    cp = np.stack([p0, c1, c2, p3])
    predicted = np.stack([b0, b1, b2, b3], 1) @ cp
    return cp, np.linalg.norm(predicted-points, axis=1)


def split_fit(points, tolerance=.01):
    cp, errors = fit_curve(points)
    chord = np.linalg.norm(cp[-1, :2]-cp[0, :2])
    u = np.linspace(0, 1, 17)
    path = np.stack([(1-u)**3, 3*u*(1-u)**2, 3*u*u*(1-u), u**3], 1) @ cp[:, :2]
    arc = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    good = errors.max() <= tolerance and arc <= 3*max(chord, 1e-8)
    if good or len(points) <= 2:
        return [cp]
    index = int(errors.argmax()) if errors.max() > tolerance else len(points)//2
    index = max(1, min(index, len(points)-2))
    return split_fit(points[:index+1], tolerance)+split_fit(points[index:], tolerance)


def curve_vector(cp, down):
    delta = cp[3, :2]-cp[0, :2]
    length = max(np.linalg.norm(delta), 1e-6)
    c1, c2 = cp[1, :2]-cp[0, :2], cp[2, :2]-cp[3, :2]
    def angle(a, b):
        if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
            return 0.
        return np.arctan2(a[0]*b[1]-a[1]*b[0], np.dot(a, b))
    t = cp[:, 2]
    gamma = [3*(t[1]-t[0]), 3*(t[2]-2*t[1]+t[0]),
             t[3]-3*t[2]+3*t[1]-t[0]]
    return [*delta, np.linalg.norm(c1)/length, np.linalg.norm(c2)/length,
            angle(delta, c1), angle(-delta, c2), *gamma, float(down)]


def validate_tolerance(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("tolerance must be finite and greater than zero")
    return value


def curve_features(strokes, tolerance=.01):
    tolerance = validate_tolerance(tolerance)
    strokes = [np.asarray(s, dtype=np.float64) for s in strokes]
    if not strokes or any(len(s) == 0 or s.shape[1] != 3 or not np.isfinite(s).all() for s in strokes):
        raise ValueError('Malformed stroke')
    xy = np.concatenate(strokes)[:, :2]
    # Isotropic y normalization following the referenced paper. Flat strokes
    # use width instead to avoid division by zero.
    span = np.ptp(xy, axis=0)
    scale = span[1] if span[1] > 1e-6 else max(span[0], 1.)
    origin = np.array([xy[0, 0], xy[:, 1].min()])
    result, previous = [], None
    for raw in strokes:
        p = raw.copy()
        p[:, :2] = (p[:, :2]-origin)/scale
        if previous is not None:
            # Explicit pen-up connector preserves the relative placement of strokes.
            a, b = previous.copy(), p[0].copy()
            a[2] = b[2] = 0.
            gap = np.stack([a, a+(b-a)/3, a+2*(b-a)/3, b])
            result.append(curve_vector(gap, False))
        length = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()
        times = np.maximum.accumulate(p[:, 2])-p[0, 2]
        p[:, 2] = times/times[-1]*length if times[-1] > 0 else np.linspace(0, length, len(p))
        for cp in split_fit(p, tolerance):
            result.append(curve_vector(cp, True))
        previous = p[-1]
    result = np.asarray(result, dtype='<f4')
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite curve features')
    return result


def convert(item, tolerance=.01):
    name, payload = item
    try:
        root = ET.fromstring(payload)
        meta = {e.attrib['type']: e.text for e in root.findall(NS+'annotation')}
        strokes = [[[float(v) for v in p.split()] for p in e.text.split(',')]
                   for e in root.findall(NS+'trace')]
        label = meta['normalizedLabel']
        tokens = tokenize(label)
        if not tokens:
            raise ValueError('Empty label')
        feat = curve_features(strokes, tolerance=tolerance)
        return (Path(name).stem, Path(name).parent.name, label, json.dumps(tokens),
                len(feat), minimum_frames(tokens), feat.tobytes()), None
    except Exception as e:
        return None, dict(name=name, error=str(e))


def archive_items(path, limit):
    counts = Counter()
    with tarfile.open(path, 'r|gz') as tar:
        for entry in tar:
            split = Path(entry.name).parent.name
            if not entry.isfile() or not entry.name.endswith('.inkml') or split not in ('train','synthetic','valid'):
                continue
            if limit and counts[split] >= limit:
                continue
            counts[split] += 1
            yield entry.name, tar.extractfile(entry).read()


def prepare(archive, out, limit=0, workers=4, tolerance=.01):
    tolerance = validate_tolerance(tolerance)
    out.mkdir(parents=True, exist_ok=True)
    dest = out/'features.sqlite'
    if dest.exists() or dest.with_suffix('.partial').exists():
        raise FileExistsError('Choose a new cache directory; existing cache will not be overwritten')
    temporary = dest.with_suffix('.partial')
    conn = sqlite3.connect(temporary)
    conn.execute('CREATE TABLE samples (id TEXT, split TEXT, label TEXT, tokens TEXT, length INTEGER, min_frames INTEGER, features BLOB)')
    counts, vocab, errors, lengths = Counter(), set(), [], []
    started = time.monotonic()
    # Bounded chunks prevent ProcessPoolExecutor.map from queuing the whole archive.
    def chunks(iterator, n=256):
        group = []
        for item in iterator:
            group.append(item)
            if len(group) == n:
                yield group
                group = []
        if group:
            yield group
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for group in chunks(archive_items(archive, limit)):
            for row, error in pool.map(partial(convert, tolerance=tolerance), group, chunksize=8):
                if error:
                    errors.append(error)
                    continue
                conn.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?)', row)
                counts[row[1]] += 1
                lengths.append(row[4])
                if row[1] in ('train','synthetic'):
                    vocab.update(json.loads(row[3]))
            conn.commit()
            if sum(counts.values()) % 5120 < 256:
                print(json.dumps(dict(processed=dict(counts), seconds=time.monotonic()-started)), flush=True)
    conn.execute('CREATE UNIQUE INDEX identity ON samples(split,id)')
    conn.execute('CREATE INDEX split_length ON samples(split,length)')
    conn.commit()
    conn.close()
    os.replace(temporary, dest)
    (out/'vocab.json').write_text(json.dumps(['<BLANK>','<UNK>']+sorted(vocab), ensure_ascii=False))
    report = dict(counts=dict(counts), errors=errors, curve_length_percentiles=
                  dict(zip(['p50','p90','p99','max'],np.percentile(lengths,[50,90,99,100]).tolist())),
                  seconds=time.monotonic()-started, archive=str(archive.resolve()), limit_per_split=limit,
                  tolerance=tolerance,
                  tolerance_space='Euclidean error in normalized x/y and stroke-relative scaled time; not a compression ratio',
                  representation=f'Approximate 10D Bezier; error tolerance {tolerance:g}; aspect preserved; pen-up connectors',
                  deviations='Endpoint constrained chord fit; no iterative Newton refinement/merge; no official preprocessing code')
    (out/'preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--tolerance',type=validate_tolerance,default=.01,
                        help='Bezier fit tolerance in normalized x/y/time (default: 0.01). Larger usually means fewer segments. Must be positive and finite.')
    args=parser.parse_args()
    prepare(args.archive,args.out,args.limit,args.workers,tolerance=args.tolerance)
