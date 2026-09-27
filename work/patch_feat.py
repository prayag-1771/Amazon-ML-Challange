import re
p = "src/features.py"
s = open(p, encoding="utf-8").read()
start = s.index("def _sets(")
end = s.index("def pair_features(")
new = '''def _jacc(a: pl.Series, b: pl.Series, sep: str) -> np.ndarray:
    """Token-set Jaccard; -1 when either side is empty."""
    d = pl.DataFrame({"a": a.str.split(sep).list.eval(pl.element().filter(pl.element() != "")),
                      "b": b.str.split(sep).list.eval(pl.element().filter(pl.element() != ""))})
    inter = pl.col("a").list.set_intersection(pl.col("b")).list.len()
    union = pl.col("a").list.set_union(pl.col("b")).list.len()
    empty = (pl.col("a").list.len() == 0) | (pl.col("b").list.len() == 0)
    return d.select(pl.when(empty).then(-1.0).otherwise(inter / union).cast(pl.Float32)).to_series().to_numpy()


def _num_matrix(nums: pl.Series, k: int = 4) -> np.ndarray:
    """First k numeric tokens as a float matrix (nan-padded); 8+ digit tokens (phones, ids) dropped."""
    lst = nums.str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars().is_between(1, 8)))
    m = np.full((len(nums), k), np.nan, dtype=np.float64)
    for j in range(k):
        m[:, j] = lst.list.get(j, null_on_oob=True).cast(pl.Float64).fill_null(np.nan).to_numpy()
    return m


def _num_feats(n1: pl.Series, n2: pl.Series):
    """House-number style features: set Jaccard, first-number match, min relative diff."""
    jac = _jacc(n1, n2, " ")
    a, b = _num_matrix(n1), _num_matrix(n2)
    first_eq = np.nanmax(np.where(np.isnan(a[:, :1, None]) | np.isnan(b[:, None, :]), np.nan,
                                  (a[:, :1, None] == b[:, None, :]) | (a[:, :, None] == b[:, None, :1])).reshape(len(a), -1)
                         if False else 0, axis=None) if False else None
    with np.errstate(invalid="ignore", divide="ignore", all="ignore"):
        eq_a0 = (a[:, :1] == b).any(axis=1)
        eq_b0 = (b[:, :1] == a).any(axis=1)
        diff = np.abs(a[:, :, None] - b[:, None, :]) / np.maximum(np.maximum(a[:, :, None], b[:, None, :]), 1)
        min_rel = np.nanmin(np.where(np.isnan(diff), np.inf, diff).reshape(len(a), -1), axis=1)
    has = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
    first_eq = np.where(has, (eq_a0 | eq_b0).astype(np.float32), -1).astype(np.float32)
    min_rel = np.where(has & np.isfinite(min_rel), min_rel, -1).astype(np.float32)
    return jac, first_eq, min_rel


'''
s = s[:start] + new + s[end:]
s = s.replace('f["city_jacc"] = _jacc(_sets(A["alpha_comps"], "|"), _sets(B["alpha_comps"], "|"))',
              'f["city_jacc"] = _jacc(A["alpha_comps"], B["alpha_comps"], "|")')
s = s.replace('f["addr_tok_jacc"] = _jacc(_sets(A["addr_core"], " "), _sets(B["addr_core"], " "))',
              'f["addr_tok_jacc"] = _jacc(A["addr_core"], B["addr_core"], " ")')
open(p, "w", encoding="utf-8").write(s)
