#!/usr/bin/env python3
from sage.all import *
import sys, time
proof.all(False)
from sage.schemes.elliptic_curves.mod_poly import classical_modular_polynomial

r'''
Optimized variant of the OneEnd/EndRing attack PoC.

Hotspot (measured by profiling): the per-node modular-polynomial step
    classical_modular_polynomial(l, j).roots()
On Sage's cached path this still evaluates a bivariate polynomial with big
INTEGER coefficients at a point of F_{p^2}, reducing those coefficients mod p
on every single call. Since p (hence F_{p^2}) is fixed for the whole attack,
we reduce Phi_l mod p exactly ONCE per prime l and store it as its list of
Y-coefficient polynomials c_i(X) in F[X]. Each subsequent evaluation is then
   f(Y) = sum_i c_i(j) * Y^i
over F, i.e. cheap arithmetic on already-reduced coefficients.

Root-finding is left to Sage's C-backed .roots() (PARI/FLINT); a Python-level
Cantor-Zassenhaus was tried and is far slower.

The Frobenius conjugate j^p is computed with .frobenius() (x -> x^p on F_{p^2}
is the nontrivial automorphism a+bT -> a-bT) instead of a generic power.
'''

################################################################
# precomputed modular-polynomial evaluation

_phi_coeffs = {}   # (l, F) -> [c_0(X), c_1(X), ...] in F[X], Phi_l = sum c_i(X) Y^i

def _prep(l, F):
    key = (int(l), F)
    cs = _phi_coeffs.get(key)
    if cs is None:
        Phi = classical_modular_polynomial(l).change_ring(F)   # reduce mod p ONCE
        RX = F['X']; Xg = RX.gen()
        Xvar, Yvar = Phi.parent().gens()
        dY = Phi.degree(Yvar)
        cs = [RX(0)] * (dY + 1)
        for (a, b), c in Phi.dict().items():
            cs[b] += c * Xg**a
        _phi_coeffs[key] = cs
    return cs

def neighbors(l, j):
    r"""j-invariants l-isogenous to j: the roots of Phi_l(j, Y) over F=parent(j)."""
    F = j.parent()
    cs = _prep(l, F)
    f = F['Y']([c(j) for c in cs])
    return f.roots(multiplicities=False)

################################################################
# generate an example instance

p = previous_prime(2**50)
print(f'{p = } ~ 2^{p.bit_length()}')

s = GF(p).one()
while s.is_square():
    s += 1
F = GF((p, 2), modulus=[-s,0,1], name='T')

E = special_supersingular_curve(F)
for _ in range(3 * p.bit_length()):
    E = choice(E.isogenies_prime_degree(2)).codomain()
print(f'attacking {E = }')

################################################################

def count(smooth, max_deg, degs=(1,)):
    ret = 1
    for l in prime_range(degs[-1], min(floor(max_deg), smooth) + 1):
        num = l + (l != degs[-1])
        ret += num * count(smooth, max_deg / l, degs + (l,))
    return ret

def isogs(smooth, max_deg, chain):
    # BFS ordering
    yield chain
    prev_deg = chain[-2] if len(chain) >= 2 else 1
    for l in prime_range(prev_deg, min(floor(max_deg), smooth) + 1):
        for j in neighbors(l, chain[-1]):
            if len(chain) >= 3 and j == chain[-3]:
                continue
            yield from isogs(smooth, max_deg / l, chain + [l, j])

################################################################

B = ceil(exp(1/3 * sqrt(log(p/2))))
print(f'attack parameter {B = }')
X = ceil(sqrt(RR(B)) * (p/2)**(1/6))
print(f'attack parameter {X = }')
print(f'estimated table size: {count(B, X)}')

t0 = time.time()
tab = dict()
for _, chain in enumerate(isogs(B, X, [E.j_invariant()])):
    if _ % 100 == 0:
        print(f'\x1b[K{len(tab) = :9}', end='\r', file=sys.stderr, flush=True)
    j = chain[-1]
    jp = j.frobenius()            # == j**p, the conjugate on F_{p^2}
    try:
        hit = tab[jp]
    except KeyError:
        pass
    else:
        break
    tab[j] = chain
else:
    print('\nno solution found (try re-running)')
    exit(1)

print(f'\nfound a solution!  elapsed = {time.time()-t0:.2f}s, table = {len(tab)}')
assert hit[-1] == j.frobenius()
sol = chain
for i in reversed(range(0, len(hit)-1, 2)):
    sol.append(hit[i+1])
    sol.append(hit[i+0].frobenius())

################################################################
# verify the solution

assert sol[0] == E.j_invariant()
assert sol[-1] == E.j_invariant()**p
ls = []
for i in range(0, len(sol)-1, 2):
    j0 = sol[i+0]
    l  = sol[i+1]
    j1 = sol[i+2]
    assert classical_modular_polynomial(l)(j0, j1) == 0
    ls.append(l)

deg = prod(ls)
print(f'{deg = } ~ p^{RR(log(deg,p))}')
print(f'{ls = }')
