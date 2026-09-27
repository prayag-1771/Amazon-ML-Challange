from pathlib import Path
p = Path("src/cross_encoder.py"); s = p.read_text(encoding="utf-8")
rep = [
('''Leakage control: train queries are split in two folds by hash. The cross-encoder trains on fold A
only; the LightGBM that consumes ce_logit trains on fold B, validates on held-out S1 queries and
predicts test - none of which the cross-encoder has seen.
"""''',
'''Leakage control: train queries are split in two folds by hash. Cross-encoder "a" trains on fold A,
cross-encoder "b" on fold B (cross-fitting). Each train query gets the logit of the model that did
not see it; validation (held-out S1) and test pairs get the mean of both logits.
"""'''),
('CE_DIR = WORK_DIR / "ce_e5s"\n', 'CE_DIRS = {"a": WORK_DIR / "ce_e5s", "b": WORK_DIR / "ce_e5s_b"}\n'),
('def train_ce(pairs: pl.DataFrame, split: str = "train", epochs: int = 1, bs: int = 128, lr: float = 5e-5):',
 'def train_ce(pairs: pl.DataFrame, out_dir, split: str = "train", epochs: int = 1, bs: int = 128, lr: float = 5e-5):'),
('''    model.save_pretrained(CE_DIR)
    tok.save_pretrained(CE_DIR)''', '''    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)'''),
('def score_ce(pairs: pl.DataFrame, split: str, model=None, bs: int = 512, chunk: int = 1_000_000) -> np.ndarray:',
 'def score_ce(pairs: pl.DataFrame, split: str, fold: str = "a", bs: int = 512, chunk: int = 1_000_000) -> np.ndarray:'),
('    model = model or AutoModelForSequenceClassification.from_pretrained(CE_DIR).cuda()',
 '    model = AutoModelForSequenceClassification.from_pretrained(CE_DIRS[fold]).cuda()'),
('''def ce_scores(split: str, force: bool = False) -> pl.DataFrame:
    """Cached ce_logit for every pair of the pruned candidate set of a split."""
    path = WORK_DIR / f"ce_{split}.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)
    cand = pl.read_parquet(WORK_DIR / f"pruned_{split}.parquet", columns=["q_row", "s1_row"])
    if split == "train":  # the cross-encoder's own training fold is never scored/used downstream
        from .prune import valid_s1_rows
        va_q = cand.join(valid_s1_rows(), on="s1_row", how="semi").select("q_row").unique()
        cand = pl.concat([cand.filter(fold_b()), cand.join(va_q, on="q_row", how="semi")]).unique(maintain_order=True)
    cand = cand.with_columns(pl.Series("ce_logit", score_ce(cand, split)))''',
'''def ce_scores(split: str, fold: str = "a", force: bool = False) -> pl.DataFrame:
    """Cached ce_logit of cross-encoder `fold` for the pruned candidates of a split.
    On train only the other fold and the validation queries are scored (never the model's own fold)."""
    path = WORK_DIR / (f"ce_{split}.parquet" if fold == "a" else f"ce_{split}_{fold}.parquet")
    if path.exists() and not force:
        return pl.read_parquet(path)
    cand = pl.read_parquet(WORK_DIR / f"pruned_{split}.parquet", columns=["q_row", "s1_row"])
    if split == "train":
        from .prune import valid_s1_rows
        va_q = cand.join(valid_s1_rows(), on="s1_row", how="semi").select("q_row").unique()
        other = cand.filter(fold_b() if fold == "a" else ~fold_b())
        cand = pl.concat([other, cand.join(va_q, on="q_row", how="semi")]).unique(maintain_order=True)
    cand = cand.with_columns(pl.Series("ce_logit", score_ce(cand, split, fold)))'''),
('''def train_stage():
    from .prune import add_labels, train_queries
    cand = pl.read_parquet(WORK_DIR / "pruned_train.parquet", columns=["q_row", "s1_row"])
    tr_q = pl.DataFrame({"q_row": train_queries(cand)}).filter(~fold_b())''',
'''def train_stage(fold: str = "a"):
    from .prune import add_labels, train_queries
    cand = pl.read_parquet(WORK_DIR / "pruned_train.parquet", columns=["q_row", "s1_row"])
    tr_q = pl.DataFrame({"q_row": train_queries(cand)}).filter(~fold_b() if fold == "a" else fold_b())'''),
('''    train_ce(pa)''', '''    train_ce(pa, CE_DIRS[fold])'''),
('''    # python -m src.cross_encoder train | score train test
    if sys.argv[1] == "train":
        train_stage()
    else:
        for split in sys.argv[2:]:
            ce_scores(split, force=True)''',
'''    # python -m src.cross_encoder train {a|b} | score {a|b} train test
    if sys.argv[1] == "train":
        train_stage(sys.argv[2])
    else:
        for split in sys.argv[3:]:
            ce_scores(split, sys.argv[2], force=True)'''),
]
for a, b in rep:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8"); print("ok")
