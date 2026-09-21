# Supersingular Frobenius-conjugate isogeny path attack

Given a supersingular elliptic curve `E / F_{p^2}`, this code searches for an
isogeny from `E` to its Frobenius conjugate `E^(p)` (the curve with j-invariant
`j(E)^p`). It does so by a **meet-in-the-middle "claw" search**: enumerate the
`j`-invariants reachable from `E` by a `B`-smooth isogeny of degree at most `X`,
and look for a `j` whose Frobenius conjugate `j^p` is also reachable. Such a
collision splices two `B`-smooth isogenies into one isogeny `E -> E^(p)`.

Two implementations are included: the **original** proof of concept and a
**fully optimized** version.

---

## Files

| file | what it is |
|------|------------|
| `attack_orig.py`   | **Original attack** (reference PoC). Depth-first enumeration of `B`-smooth isogeny chains; for each node it computes neighbours as the roots of `classical_modular_polynomial(l, j)`. Single instance: prints a solution or "re-run" if the one table has no claw. |
| `attack_opt.py`    | **Intermediate optimization.** Same enumeration, but reduces `Phi_l mod p` once per prime and evaluates the cached `V`-coefficient polynomials (avoids re-reducing big integer coefficients every call). Root-finding is still Sage's `.roots()`. |
| `isogeny_list.py`  | **Core library** for the optimized attack: modular-polynomial cache, **batched multipoint evaluation** of `Phi_l` (subproduct / remainder tree), the layered **parent-pointer list** `L(E,X,B)`, edge enumeration via **PARI fast root-finding**, and path reconstruction. Importing it runs nothing. |
| `attack_batched.py`| **Full optimized attack.** Layered construction of `L(E,X,B)` with batched `Phi_l` evaluation + PARI fast randomized root-finding, an integrated meet-in-the-middle collision test, automatic **rerandomization** until a claw is found, and verification of the resulting `E -> E^(p)` isogeny. |
| `smooth_sigma.py`  | **Heuristic success-probability tool.** Computes `Sigma_{B,X} = sum_{n in D_{B,X}} sqrt(n)` and `Pr[claw] ~ 1 - exp(-sqrt(12/p) * Sigma_{B,X})`, the per-instance probability that a random curve has a claw (so `1/Pr` is the expected number of rerandomizations). |

---

## Requirements

- **SageMath** (tested with 10.7). Everything runs inside Sage.
- **PARI/GP** — bundled with Sage. The optimized attack uses `pari.polmodular`
  to build the classical modular polynomials `Phi_l mod p` and PARI's
  `polrootsff` / `factorff` for fast root-finding over `F_{p^2}`. No optional
  Kohel modular-polynomial database is required.
- `smooth_sigma.py` is plain Python (no Sage needed): `python3 smooth_sigma.py`.

All scripts set `proof.all(False)` and build `F_{p^2}` as
`GF((p,2), modulus=[-s,0,1])` with `s` a quadratic non-residue mod `p`, matching
the original PoC.

---

## How to run

### Original attack (single instance)
```sh
sage attack_orig.py
```
Generates a random supersingular `E` at `p = previousprime(2^40)`, uses the
default parameters, and either prints a verified solution or
`no solution found (try re-running)`. Re-run to draw a fresh instance.

### Intermediate optimization
```sh
sage attack_opt.py
```
Same behaviour as the original but faster per node.

### Full optimized attack (auto-rerandomizing)
```sh
sage attack_batched.py            # default p ~ 2^40
sage attack_batched.py 32         # choose the bit-size of p
```
Draws random supersingular instances and **keeps rerandomizing until one has a
claw**, then reconstructs and verifies the `E -> E^(p)` isogeny and prints its
degree and the list of prime steps. Example output:
```
seed 0: SOLVED  |L|=6170  time=0.89s
  degree = 1875 ~ p^0.340
  primes = [3, 5, 5, 5, 5]
```

### Success-probability / rerandomization estimate
```sh
python3 smooth_sigma.py
```
or from Sage:
```python
from smooth_sigma import sigma_BX, mu_BX
mu, P = mu_BX(p, B, X)     # P = heuristic Pr[claw];  1/P ~ expected rerandomizations
```

---

## Parameters

Both attacks use the same defaults (subexponentially balanced):
```
B = ceil(exp( sqrt(log(p/2)) / 3 ))        # smoothness bound
X = ceil( sqrt(B) * (p/2)^(1/6) )                # meet-in-the-middle degree bound (original)
```
`attack_batched.run_attack(j0, B, X, ...)` also accepts:
- `edge_mode`  : `"distinct"` (default here) or `"multiplicity"`,
- `compat`     : `"legacy"` (reproduces the original's pruning) or `"strict"`,
- `crossover`  : frontier size below which evaluation is done point-by-point
  instead of via the subproduct tree (default 32),
- `root_backend`: `"pari"` (fast, default) or `"sage"` (reference),
- `max_entries`: optional cap on the table size.

---

## What "optimized" means

The optimized attack replaces the per-node work of the original with two changes
that do **not** alter the output:

1. **Batched multipoint evaluation.** Instead of evaluating `Phi_l` at one
   `j`-invariant per node, a whole `(prime, valuation)` layer is evaluated at
   once by a shared subproduct/remainder tree, in
   `O(l * M(m+l) * log(m+l))` field operations for a frontier of `m` curves.
2. **Fast randomized root-finding.** Neighbours are the roots of
   `Phi_l(j, V)` over `F_{p^2}`, computed with PARI's `polrootsff`
   (Cantor–Zassenhaus, expected `O(M(l) log l log(l p))` field ops) instead of
   Sage's generic `.roots()`.

On these parameters root-finding dominates, so (2) is the bigger win; together
they give roughly a 2.5x end-to-end speedup at `p ~ 2^40`–`2^50`, with output
identical to the original.

---

## Output format

A solution is a chain in the format used by the original verifier:
```
[ j0, l1, j1, l2, j2, ..., ln, jn ]
```
with `j0 = j(E)`, `jn = j(E)^p`, each `li` a prime `<= B`, and
`classical_modular_polynomial(li)(j_{i-1}, j_i) = 0`. The product of the `li` is
the degree of the `E -> E^(p)` isogeny (`~ p^{1/3}` on average).

---

## The `smooth_sigma.py` success-probability heuristic

`smooth_sigma.py` estimates the **per-instance probability that a random curve
has a claw**, i.e. how many rerandomizations the optimized attack is expected to
need (`1 / Pr[claw]`).

**Definitions it implements**

- `S(X,B) = { m <= X : P^+(m) <= B }` — the `B`-smooth integers up to `X`
  (`P^+(m)` is the largest prime factor of `m`, `P^+(1)=1`).
- `D_{B,X} = { a*b : a,b in S(X,B) }` — the **admissible meet-in-the-middle
  degrees**: `n` is admissible iff a `B`-smooth `n`-isogeny splits into two
  `B`-smooth isogenies each of degree `<= X`.
- `Sigma_{B,X} = sum_{n in D_{B,X}} sqrt(n)` — the weighted admissible-degree
  mass.
- `mu_{B,X} = sqrt(12/p) * Sigma_{B,X}` — heuristic expected number of `B`-smooth
  isogenies `E -> E^(p)` of admissible degree, modelled as a Poisson mean.
- `Pr[claw] ~ 1 - exp(-mu_{B,X})` — the per-instance success probability;
  `1/Pr[claw]` is the expected number of rerandomizations.

**Functions**

- `smooth_numbers(X, B)` — the list `S(X,B)` of `B`-smooth integers `<= X`.
- `sigma_BX(B, X)` — returns `(Sigma, |S|, |D|)` from the direct product-set
  definition of `D_{B,X}`; `O(|S|^2)` time, `O(|D|)` space.
- `sigma_BX_fast(B, X)` — the same `Sigma`, obtained by enumerating the
  `B`-smooth `n <= X^2` and keeping those that split (a divisor in `[n/X, X]`);
  agrees with `sigma_BX` to floating precision.
- `mu_BX(p, B, X)` — returns `(mu, Pr[claw])`.

**Usage**

```sh
python3 smooth_sigma.py            # self-test: reproduces reference Sigma values
```
```python
from smooth_sigma import sigma_BX, mu_BX
Sigma, nS, nD = sigma_BX(B, X)     # weighted admissible-degree mass, |S|, |D|
mu, P = mu_BX(p, B, X)             # P = heuristic Pr[claw]; 1/P = expected rerandomizations
```

**Accuracy** — the functional form was checked against measured success rates
over `p = 2^28 .. 2^50` and `B = 5 .. 17`: it tracks the data well, with the
heuristic slightly under-counting (the true mean is about `1.3x` the estimate),
so treat `Pr[claw]` as an order-of-magnitude / lower-bound guide. Pure Python
(no Sage required); `sqrt` is double precision.

## Notes

- The **original** attacks one instance and asks you to re-run if that instance
  has no claw. The **optimized** attack rerandomizes automatically; the expected
  number of rerandomizations is `1/Pr[claw]` (estimate it with `smooth_sigma`).
- Both attacks are heuristic and randomized; running times and the degree of the
  recovered isogeny vary from run to run.
