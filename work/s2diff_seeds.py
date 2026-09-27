"""Seed-robustness of +ndiff3 (seeds 17, 27) vs base."""
exec(open("../work/s2diff.py").read().split("cv(S.F, \"base   \")")[0])
import itertools
for sd in (17, 27):
    S.P = {**S.P, "seed": sd}
    cv(S.F, f"base s{sd}   ")
    cv(S.F + ["ndiff","nd3","nratio"], f"+ndiff3 s{sd}")
