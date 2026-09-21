#!/usr/bin/env python3
r"""
Listing B-smooth isogenies with batched (multipoint) modular-polynomial evaluation.

This module implements the algorithm

    Listing B-smooth isogenies with batched evaluation

which, from a supersingular curve E/F_{p^2} and parameters B (smoothness bound)
and X (degree bound), builds an indexed *parent-pointer* list

    L = [ Entry(j, pi, d), ... ]      (1-indexed)

representing L(E, X, B).  Each entry stores only:

    j   : j-invariant of the codomain of the isogeny  E -> E_t
    pi  : index (into L) of the parent entry           (0 for the root)
    d   : degree of the isogeny  E -> E_t

A full isogeny chain [j0, l1, j1, ..., ln, jt] to any entry is reconstructed on
demand by walking parent pointers to the root (see reconstruct_chain).  This is
the memory win over attack_orig.py / attack_opt.py, which store a full chain in
every table slot.

The core optimisation is BatchEvaluate: instead of evaluating the modular
polynomial Phi_l at each frontier j-invariant separately (attack_opt.py's
`neighbors`), a whole layer (l, i) is evaluated at once.  Writing

    Phi_l(U, V) = sum_{k=0}^{l+1} f_{l,k}(U) V^k,   deg f_{l,k} <= l+1,

each coefficient polynomial f_{l,k} is evaluated at ALL m frontier points
j_1,...,j_m simultaneously by a single subproduct-tree / remainder-tree
multipoint evaluation (von zur Gathen & Gerhard, Modern Computer Algebra,
Cor. 10.8).  This costs

    O( l * M(m + l) * log(m + l) )

field operations per layer, versus the naive O(m * l * (l+1)) of evaluating
each of the l+2 coefficient polynomials at each of the m points by Horner.

Conventions match the existing scripts:  `from sage.all import *`,
`proof.all(False)`, F = GF((p,2), modulus=[-s,0,1], name='T').
"""

from sage.all import *
proof.all(False)
from sage.schemes.elliptic_curves.mod_poly import classical_modular_polynomial

from collections import namedtuple

# An entry of the parent-pointer list L.  j: F_{p^2} element, pi: parent index
# (0 for the root), d: degree of E -> codomain.
Entry = namedtuple("Entry", ["j", "pi", "d"])


# ----------------------------------------------------------------------------
# Modular-polynomial cache: reduce Phi_l mod p ONCE per (l, field) and keep it
# as the list of V-coefficient polynomials f_{l,k}(U) in F[U].
# ----------------------------------------------------------------------------

_phi_coeffs = {}   # (l, F) -> [f_{l,0}(U), ..., f_{l,l+1}(U)] in F[U]


def phi_coeffs(l, F):
    r"""
    Return [f_{l,0}, ..., f_{l,l+1}] in F[U] with
        Phi_l(U, V) = sum_k f_{l,k}(U) * V^k.
    Computed once per (l, F) and cached.
    """
    l = int(l)
    if not is_prime(l):
        raise ValueError(f"level l={l} is not prime")
    key = (l, F)
    cs = _phi_coeffs.get(key)
    if cs is None:
        Phi = classical_modular_polynomial(l).change_ring(F)   # reduce mod p once
        RU = F["U"]
        Ug = RU.gen()
        Uvar, Vvar = Phi.parent().gens()
        dV = Phi.degree(Vvar)
        cs = [RU(0)] * (dV + 1)
        for (a, b), c in Phi.dict().items():
            cs[b] += c * Ug**a
        _phi_coeffs[key] = cs
    return cs


# ----------------------------------------------------------------------------
# Multipoint evaluation via subproduct tree + remainder tree.
# ----------------------------------------------------------------------------

def build_subproduct_tree(RU, points):
    r"""
    Build a subproduct tree over the ring RU = F[U] for the given points.

    Leaves are the linear factors (U - u_i) in the order given.  Each internal
    node holds the product of its two children.  Returned as a list of levels,
    level 0 = leaves, last level = single root node (the full product).
    """
    U = RU.gen()
    level = [U - RU(u) for u in points]
    tree = [level]
    while len(level) > 1:
        nxt = []
        for k in range(0, len(level) - 1, 2):
            nxt.append(level[k] * level[k + 1])
        if len(level) % 2 == 1:
            nxt.append(level[-1])      # carry the odd node up unchanged
        tree.append(nxt)
        level = nxt
    return tree


def multipoint_eval_tree(f, tree):
    r"""
    Evaluate f at all points of a prebuilt subproduct `tree`, returning the list
    of values [f(u_0), ..., f(u_{m-1})] in the leaf order of the tree.

    Standard remainder-tree descent: reduce f modulo the node polynomial at each
    node going down; at a leaf (U - u_i) the remainder is the constant f(u_i).
    """
    m = len(tree[0])
    if m == 0:
        return []
    if m == 1:
        # single leaf (U - u_0); f mod (U - u_0) = f(u_0)
        return [(f % tree[0][0]).constant_coefficient()]

    top = len(tree) - 1
    # remainder at the root
    rem = {(top, 0): f % tree[top][0]}
    for lev in range(top, 0, -1):
        nodes = tree[lev]
        child = tree[lev - 1]
        for idx in range(len(nodes)):
            r = rem[(lev, idx)]
            lc = 2 * idx
            rc = 2 * idx + 1
            if rc < len(child):
                rem[(lev - 1, lc)] = r % child[lc]
                rem[(lev - 1, rc)] = r % child[rc]
            else:
                # odd carried node: same polynomial one level down
                rem[(lev - 1, lc)] = r
    leaves = tree[0]
    return [rem[(0, i)].constant_coefficient() for i in range(len(leaves))]


def multipoint_eval_direct(f, points):
    r"""Direct per-point evaluation (Horner via Sage), the crossover fallback."""
    return [f(u) for u in points]


def batch_evaluate(l, coeffs, points, crossover=32):
    r"""
    BatchEvaluate: for a layer with frontier j-invariants `points` and the
    coefficient polynomials `coeffs` = [f_{l,0},...,f_{l,l+1}] of Phi_l, return

        [ Phi_l(points[t], V) in F[V]  for each t ]

    computed by a single subproduct tree shared across all l+2 coefficient
    polynomials.  Falls back to direct evaluation when m <= crossover.

    Returns a list of univariate polynomials in F[V], one per frontier point,
    in the same order as `points`.
    """
    m = len(points)
    if m == 0:
        return []
    RU = coeffs[0].parent()
    F = RU.base_ring()
    RV = F["V"]

    if m <= crossover:
        # per-coefficient direct evaluation
        cols = [multipoint_eval_direct(fk, points) for fk in coeffs]
    else:
        tree = build_subproduct_tree(RU, points)
        cols = [multipoint_eval_tree(fk, tree) for fk in coeffs]

    # cols[k][t] = f_{l,k}(points[t]); assemble Phi_l(points[t], V)
    out = []
    for t in range(m):
        out.append(RV([cols[k][t] for k in range(len(coeffs))]))
    return out


# ----------------------------------------------------------------------------
# Edge enumeration from Phi_l(j_t, V).
# ----------------------------------------------------------------------------

def _roots_pari(poly, mode):
    r"""
    Roots of `poly` over F_{p^2} via PARI's fast randomized factorization
    (expected O(M(l) log l log(l p)) field ops), which is ~3-4x faster than
    Sage's generic .roots() for these polynomials.  Only F_{p^2}-rational roots
    (degree-1 factors) are returned, which is exactly what neighbour enumeration
    needs.
    """
    F = poly.base_ring()
    if mode == "distinct":
        return [F(r) for r in poly.__pari__().polrootsff()]
    # multiplicity: factor and read exponents of the linear factors
    fac = poly.__pari__().factorff()
    out = []
    for g, e in zip(fac[0], fac[1]):
        if g.poldegree() == 1:
            r = F(-g.polcoeff(0) / g.polcoeff(1))
            out.extend([r] * int(e))
    return out


def edges_from_poly(poly, mode="multiplicity", backend="pari"):
    r"""
    Roots of Phi_l(j_t, V) over F, i.e. the j-invariants of the l-isogenous
    neighbours.  `mode`:
        "multiplicity" -> each root repeated according to its multiplicity
        "distinct"     -> each distinct root once (matches the existing scripts)
    `backend`:
        "pari"  -> PARI fast randomized factorization (default, fastest)
        "sage"  -> Sage's generic .roots() (reference / validation)
    Returns a list of F elements (codomain j-invariants).
    """
    if poly.is_zero():
        raise ValueError("Phi_l(j_t, V) is the zero polynomial")
    if backend == "pari":
        return _roots_pari(poly, mode)
    # ---- Sage reference backend ----
    if mode == "distinct":
        return [r for (r, _mult) in poly.roots()]
    elif mode == "multiplicity":
        out = []
        for (r, mult) in poly.roots():
            out.extend([r] * int(mult))
        return out
    else:
        raise ValueError(f"unknown edge-counting mode {mode!r}")


# ----------------------------------------------------------------------------
# The main algorithm: build the parent-pointer list L.
# ----------------------------------------------------------------------------

def list_smooth_isogenies(j0, B, X,
                          edge_mode="multiplicity",
                          compat="strict",
                          crossover=32,
                          max_entries=None):
    r"""
    Build L(E, X, B) as an indexed parent-pointer list of Entry(j, pi, d).

    Arguments:
        j0          : j-invariant j(E) in F = F_{p^2}
        B           : smoothness bound (prime factors of every degree <= B)
        X           : degree bound (every degree <= X)
        edge_mode   : "multiplicity" (default) or "distinct"
        compat      : backtracking convention
                        "strict" -> suppress the single edge that reverses the
                                    final edge of the path (only when l equals
                                    that final edge's prime)
                        "legacy" -> suppress every edge back to the parent's
                                    j-invariant, reproducing isogs() in
                                    attack_orig.py / attack_opt.py
        crossover   : frontier size at/below which BatchEvaluate uses direct
                      per-point evaluation instead of the subproduct tree
        max_entries : optional cap on |L|; enumeration stops before exceeding it

    Returns the list L (0-based Python list; L[0] is the root, "index 1").
    """
    B = Integer(B)
    X = Integer(X)
    if B < 2:
        raise ValueError(f"B={B} must be >= 2")
    if X < 1:
        raise ValueError(f"X={X} must be >= 1")

    F = j0.parent()

    # L[0] is the root entry (the "index 1" of the pseudocode).
    L = [Entry(F(j0), 0, Integer(1))]

    def _stop_reached():
        return max_entries is not None and len(L) >= max_entries

    for l in prime_range(2, int(B) + 1):
        l = Integer(l)
        if l > X:
            break
        cs = phi_coeffs(l, F)

        imax = 0
        while l ** (imax + 1) <= X:
            imax += 1
        # imax = floor(log_l X); layers i = 1..imax

        for i in range(1, imax + 1):
            # Frontier F_{l,i}: entries t with v_l(d_t) = i-1 and l*d_t <= X.
            # (Snapshot |L| now; entries appended in this layer are NOT extended
            #  until layer (l, i+1) by the valuation predicate.)
            frontier = [t for t in range(len(L))
                        if L[t].d.valuation(l) == i - 1 and l * L[t].d <= X]
            if not frontier:
                continue

            pts = [L[t].j for t in frontier]
            polys = batch_evaluate(l, cs, pts, crossover=crossover)

            for fi, t in enumerate(frontier):
                if _stop_reached():
                    return L
                ent = L[t]
                nbrs = edges_from_poly(polys[fi], mode=edge_mode)

                if t == 0:
                    # this entry IS the root (index 0): no final edge to reverse
                    kept = nbrs
                else:
                    parent_j = L[ent.pi].j
                    fep = ent.d // L[ent.pi].d      # final-edge prime
                    if compat == "legacy":
                        kept = [jn for jn in nbrs if jn != parent_j]
                    elif compat == "strict":
                        if l == fep:
                            kept = []
                            removed = False
                            for jn in nbrs:
                                if (not removed) and jn == parent_j:
                                    removed = True          # drop exactly one
                                    continue
                                kept.append(jn)
                            if not removed:
                                raise RuntimeError(
                                    f"strict: no reversing edge at t={t}, l={l}")
                        else:
                            kept = nbrs
                    else:
                        raise ValueError(f"unknown compat mode {compat!r}")

                for jn in kept:
                    if _stop_reached():
                        return L
                    L.append(Entry(jn, t, l * ent.d))

    return L


# ----------------------------------------------------------------------------
# Path reconstruction and j-invariant index (downstream interface).
# ----------------------------------------------------------------------------

def reconstruct_chain(L, t):
    r"""
    Return the chain [j_root, l_1, j_1, ..., l_n, j_t] to entry index t
    (0-based).  Position 0,2,... hold j-invariants; positions 1,3,... hold the
    prime degrees.  Matches the chain format of attack_orig.py / attack_opt.py.
    """
    if not (0 <= t < len(L)):
        raise IndexError(f"index {t} out of range 1..{len(L)}")
    # walk parent pointers from t up to the root (index 0), then reverse
    seq = []
    cur = t
    while True:
        seq.append(cur)
        if cur == 0:
            break
        cur = L[cur].pi
    seq.reverse()                    # root ... t
    chain = [L[seq[0]].j]
    for a, b in zip(seq, seq[1:]):
        l = L[b].d // L[a].d
        chain.append(Integer(l))
        chain.append(L[b].j)
    return chain


def jinvariant_index(L):
    r"""Map each distinct j-invariant present in L to the list of its indices."""
    idx = {}
    for t, ent in enumerate(L):
        idx.setdefault(ent.j, []).append(t)
    return idx


# ----------------------------------------------------------------------------
# Reference (naive) lister: same layer loop, per-point evaluation only.
# ----------------------------------------------------------------------------

def list_smooth_isogenies_naive(j0, B, X,
                                edge_mode="multiplicity",
                                compat="strict",
                                max_entries=None):
    r"""Identical semantics to list_smooth_isogenies but forces direct
    per-point evaluation (crossover = +inf), for equivalence checking."""
    return list_smooth_isogenies(j0, B, X, edge_mode=edge_mode, compat=compat,
                                 crossover=10**18, max_entries=max_entries)


# ----------------------------------------------------------------------------
# Self-test when run directly.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    p = previous_prime(2**40)
    print(f"p = {p} ~ 2^{p.bit_length()}")
    s = GF(p).one()
    while s.is_square():
        s += 1
    F = GF((p, 2), modulus=[-s, 0, 1], name="T")

    E = special_supersingular_curve(F)
    for _ in range(3 * p.bit_length()):
        E = choice(E.isogenies_prime_degree(2)).codomain()
    j0 = E.j_invariant()
    print(f"attacking E with j = {j0}")

    B = ceil(exp(1/3 * sqrt(log(p/2))))
    X = ceil(sqrt(RR(B)) * (p/2)**(1/6))
    print(f"B = {B}, X = {X}")

    t0 = time.time()
    L = list_smooth_isogenies(j0, B, X, edge_mode="distinct", compat="legacy")
    t1 = time.time()
    print(f"batched : |L| = {len(L)}  in {t1-t0:.2f}s")

    t0 = time.time()
    Ln = list_smooth_isogenies_naive(j0, B, X, edge_mode="distinct", compat="legacy")
    t1 = time.time()
    print(f"naive   : |L| = {len(Ln)}  in {t1-t0:.2f}s")

    # structural checks
    for t in range(1, len(L)):
        ent = L[t]
        assert ent.pi < t, f"pi >= t at {t}"
        l = ent.d // L[ent.pi].d
        assert is_prime(l) and l <= B, f"bad prime at {t}"
        assert ent.d <= X
        assert classical_modular_polynomial(l)(L[ent.pi].j, ent.j) == 0, \
            f"Phi_l check failed at {t}"

    # a reconstructed chain verifies as a valid isogeny path
    if len(L) > 1:
        c = reconstruct_chain(L, len(L) - 1)
        assert c[0] == j0
        for k in range(0, len(c) - 1, 2):
            assert classical_modular_polynomial(c[k+1])(c[k], c[k+2]) == 0
        print(f"sample chain to index {len(L)-1}: degree "
              f"{prod(c[1::2]) if len(c) > 1 else 1}")

    # equivalence: same (j, d) multiset content
    key = lambda M: sorted((ent.d, str(ent.j)) for ent in M)
    assert key(L) == key(Ln), "batched and naive disagree!"
    print("OK: structural checks pass and batched == naive")
