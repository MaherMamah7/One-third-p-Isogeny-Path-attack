#!/usr/bin/env python3
r"""
Full supersingular "path to the Frobenius conjugate" attack, using the batched
(multipoint) modular-polynomial evaluation from isogeny_list.py in place of the
per-node evaluation used by attack_orig.py / attack_opt.py.

Attack (meet in the middle), identical in spirit to attack_orig.py:
    - enumerate j-invariants reachable from E by a B-smooth isogeny of degree <= X,
    - as each new j is reached, test whether its Frobenius conjugate j^p was
      already reached; a hit gives paths E -> j and E -> j^p, which splice into
      a single isogeny  E -> E^(p).

The enumeration here is the layered parent-pointer construction L(E, X, B): one
batched evaluation of Phi_l per (prime, valuation) layer over the whole frontier,
instead of one evaluation per node.  The collision test is O(1) per entry and is
identical to the per-node cost of the existing scripts, so wall-clock differences
between `crossover=<small>` (batched) and `crossover=+inf` (per-point) isolate
the contribution of batched evaluation.
"""

from sage.all import *
import time
proof.all(False)
from sage.schemes.elliptic_curves.mod_poly import classical_modular_polynomial

from isogeny_list import (Entry, phi_coeffs, batch_evaluate, edges_from_poly,
                          reconstruct_chain)

NAIVE_CROSSOVER = 10**18   # force per-point evaluation everywhere


def _build_solution(L, cur, hit):
    r"""
    Splice E -> j (index `cur`) with E -> j^p (index `hit`) into a chain
    E -> E^(p) in the [j0, l1, j1, ...] format used by the existing verifier.
    """
    c1 = reconstruct_chain(L, cur)      # E -> j
    c2 = reconstruct_chain(L, hit)      # E -> j^p
    sol = list(c1)
    # walk c2 backwards, conjugating each j (mirrors attack_opt.py)
    for i in reversed(range(0, len(c2) - 1, 2)):
        sol.append(c2[i + 1])                 # prime l
        sol.append(c2[i].frobenius())         # conjugate of the j-invariant
    return sol


def run_attack(j0, B, X,
               edge_mode="distinct", compat="legacy",
               crossover=32, root_backend="pari",
               max_entries=None, on_progress=None):
    r"""
    Run the layered MITM attack from the curve with j-invariant j0.

    Returns (solution_or_None, stats).  `solution` is the [j0, l, j1, ...] chain
    from E to E^(p) if a collision is found, else None (re-run on a fresh
    instance).  `stats` records entries built, layers, frontier points evaluated,
    time spent in evaluation, and total time.
    """
    F = j0.parent()
    B = Integer(B); X = Integer(X)

    L = [Entry(F(j0), 0, Integer(1))]
    seen = {}                                   # j -> first index reaching it
    stats = {"entries": 1, "layers": 0, "points_evaluated": 0,
             "eval_time": 0.0, "total_time": 0.0}
    t_start = time.time()

    def record_or_hit(idx):
        j = L[idx].j
        h = seen.get(j.frobenius())             # was j^p reached before?
        if h is not None:
            return h
        if j not in seen:
            seen[j] = idx
        return None

    hit = record_or_hit(0)
    if hit is not None:
        stats["total_time"] = time.time() - t_start
        return _build_solution(L, 0, hit), stats

    for l in prime_range(2, int(B) + 1):
        l = Integer(l)
        if l > X:
            break
        cs = phi_coeffs(l, F)

        imax = 0
        while l ** (imax + 1) <= X:
            imax += 1

        for i in range(1, imax + 1):
            frontier = [t for t in range(len(L))
                        if L[t].d.valuation(l) == i - 1 and l * L[t].d <= X]
            if not frontier:
                continue
            stats["layers"] += 1
            pts = [L[t].j for t in frontier]

            t0 = time.time()
            polys = batch_evaluate(l, cs, pts, crossover=crossover)
            stats["eval_time"] += time.time() - t0
            stats["points_evaluated"] += len(pts)

            for fi, t in enumerate(frontier):
                ent = L[t]
                nbrs = edges_from_poly(polys[fi], mode=edge_mode,
                                       backend=root_backend)

                if t == 0:            # this entry IS the root (index 0)
                    kept = nbrs
                else:
                    parent_j = L[ent.pi].j
                    fep = ent.d // L[ent.pi].d
                    if compat == "legacy":
                        kept = [jn for jn in nbrs if jn != parent_j]
                    else:   # strict
                        if l == fep:
                            kept = []
                            removed = False
                            for jn in nbrs:
                                if (not removed) and jn == parent_j:
                                    removed = True
                                    continue
                                kept.append(jn)
                        else:
                            kept = nbrs

                for jn in kept:
                    if max_entries is not None and len(L) >= max_entries:
                        stats["total_time"] = time.time() - t_start
                        return None, stats
                    idx = len(L)
                    L.append(Entry(jn, t, l * ent.d))
                    stats["entries"] += 1
                    hit = record_or_hit(idx)
                    if hit is not None:
                        stats["total_time"] = time.time() - t_start
                        return _build_solution(L, idx, hit), stats

            if on_progress is not None:
                on_progress(l, i, len(frontier), len(L))

    stats["total_time"] = time.time() - t_start
    return None, stats


def verify_solution(sol, j0, p):
    r"""Check sol is a valid isogeny chain E -> E^(p); return list of primes."""
    assert sol[0] == j0, "chain does not start at j(E)"
    assert sol[-1] == j0 ** p, "chain does not end at j(E)^p"
    ls = []
    for i in range(0, len(sol) - 1, 2):
        ja, l, jb = sol[i], sol[i + 1], sol[i + 2]
        assert classical_modular_polynomial(l)(ja, jb) == 0, \
            f"broken edge at position {i}"
        ls.append(Integer(l))
    return ls


def make_instance(bits=40, seed=None):
    r"""Build (p, F, j0, B, X) exactly as the existing scripts do."""
    p = previous_prime(2 ** bits)
    s = GF(p).one()
    while s.is_square():
        s += 1
    F = GF((p, 2), modulus=[-s, 0, 1], name="T")
    if seed is not None:
        set_random_seed(seed)
    E = special_supersingular_curve(F)
    for _ in range(3 * p.bit_length()):
        E = choice(E.isogenies_prime_degree(2)).codomain()
    j0 = E.j_invariant()
    B = ceil(exp(1 / 3 * sqrt(log(p / 2))))
    X = ceil(sqrt(RR(B)) * (p / 2) ** (1 / 6))
    return p, F, j0, B, X


if __name__ == "__main__":
    import sys
    bits = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    print(f"searching for an instance with a solution (bits={bits}) ...")
    for seed in range(200):
        p, F, j0, B, X = make_instance(bits=bits, seed=seed)
        sol, stats = run_attack(j0, B, X, crossover=32)
        if sol is not None:
            print(f"seed {seed}: SOLVED  |L|={stats['entries']}  "
                  f"time={stats['total_time']:.2f}s")
            ls = verify_solution(sol, j0, p)
            deg = prod(ls)
            print(f"  degree = {deg} ~ p^{RR(log(deg, p)):.3f}")
            print(f"  primes = {ls}")
            break
        else:
            print(f"seed {seed}: no collision (|L|={stats['entries']}, "
                  f"{stats['total_time']:.1f}s) - retrying")
    else:
        print("no solution across the seeds tried; re-run with more seeds")
