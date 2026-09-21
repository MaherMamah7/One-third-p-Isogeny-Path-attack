#!/usr/bin/env python3
r"""
Compute  Sigma_{B,X} = sum_{n in D_{B,X}} sqrt(n),  where
    S(X,B)   = { m <= X : P^+(m) <= B }        (B-smooth integers up to X)
    D_{B,X}  = { a*b : a,b in S(X,B) }          (distinct admissible MITM degrees)
and the Frobenius-claw mass  mu_{B,X} = sqrt(12/p) * Sigma_{B,X}.
"""
import math

def primes_up_to(B):
    """primes <= B via a simple sieve."""
    if B < 2: return []
    sieve = [True]*(B+1); sieve[0]=sieve[1]=False
    for i in range(2, int(B**0.5)+1):
        if sieve[i]:
            for j in range(i*i, B+1, i): sieve[j]=False
    return [i for i in range(2, B+1) if sieve[i]]


def smooth_numbers(X, B):
    r"""S(X,B): all B-smooth integers <= X (each once, includes 1). O(|S|) space."""
    S = [1]
    for p in primes_up_to(B):
        A = []
        for m in S:
            q = p
            while m * q <= X:
                A.append(m * q)
                q *= p
        S.extend(A)
    return S


def sigma_BX(B, X):
    r"""
    Sigma_{B,X} = sum over the DISTINCT product set D_{B,X} of sqrt(n).
    Direct definition: form all products a*b (a,b in S), dedup, sum sqrt.
    Time O(|S|^2), space O(|D|).  |S| = Psi(X,B),  |D| <= Psi(X^2,B).
    """
    S = smooth_numbers(X, B)
    S.sort()
    D = set()
    for i, a in enumerate(S):
        for b in S[i:]:            # b >= a: each unordered product once
            D.add(a * b)
    return math.fsum(math.sqrt(n) for n in D), len(S), len(D)


def sigma_BX_fast(B, X):
    r"""
    Same value, generated without the |S|^2 product loop: enumerate the B-smooth
    integers n <= X^2 and keep those that SPLIT, i.e. that have a divisor in
    [ceil(n/X), X] (equivalently n = a*b with a,b <= X).  For n <= X this is
    automatic (a=n,b=1).  Uses the sorted S(X,B) and, for each smooth n <= X^2,
    a two-pointer/binary check against divisors in the window.
    """
    S = sorted(smooth_numbers(X, B))            # B-smooth <= X
    Sset = set(S)
    # B-smooth <= X^2:
    T = sorted(smooth_numbers(X * X, B))
    Sig = 0.0
    import bisect
    for n in T:
        if n <= X:
            Sig += math.sqrt(n); continue
        # need a divisor a of n with n/X <= a <= X (then b=n/a is B-smooth <= X)
        lo = -(-n // X)                          # ceil(n/X)
        # scan candidate divisors a in S within [lo, X]
        L = bisect.bisect_left(S, lo)
        R = bisect.bisect_right(S, X)
        ok = False
        for a in S[L:R]:
            if n % a == 0:
                ok = True; break
        if ok:
            Sig += math.sqrt(n)
    return Sig


def mu_BX(p, B, X):
    r"""Frobenius-claw mean mu = sqrt(12/p) * Sigma_{B,X}, and P = 1 - exp(-mu)."""
    Sig, _, _ = sigma_BX(B, X)
    mu = math.sqrt(12.0 / float(p)) * Sig
    return mu, 1.0 - math.exp(-mu)


if __name__ == "__main__":
    # reproduce Sigma values measured earlier from the grid (Sig_D column)
    checks = [
        # (B, X, expected_Sigma)
        (5,  51,  2233.9),
        (7,  60,  5487.4),
        (5,  81,  4846.3),
        (7,  96, 11025.8),
        (11, 76, 11977.2),
    ]
    print(f"{'B':>3} {'X':>5} {'|S|':>5} {'|D|':>7} {'Sigma':>12} {'expected':>10} {'fast==direct':>13}")
    for B, X, exp in checks:
        Sig, nS, nD = sigma_BX(B, X)
        Sig2 = sigma_BX_fast(B, X)
        agree = abs(Sig - Sig2) < 1e-6 * Sig
        print(f"{B:>3} {X:>5} {nS:>5} {nD:>7} {Sig:>12.1f} {exp:>10.1f} {str(agree):>13}")
