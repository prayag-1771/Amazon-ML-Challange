"""Missing-value profile per split x country x source (raw + normalised fields)."""
import polars as pl

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(30)
W = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/"
NULLTOK = r"(?i)(^|,)\s*(null|none|n/?a|nan|<null>|-)\s*(,|$)"

rows = []
for sp in ("train", "test"):
    for s in (1, 2, 3):
        raw = pl.scan_parquet(W + f"raw_{sp}_s{s}.parquet").select("country", "business_name", "business_address")
        nrm = pl.scan_parquet(W + f"norm_{sp}_s{s}.parquet").select("state", "nums", "alpha_comps", "addr_empty", "name_core", "legal")
        d = pl.concat([raw, nrm], how="horizontal")
        a = pl.col("business_address").str.strip_chars()
        n = pl.col("business_name").str.strip_chars()
        r = d.group_by("country").agg(
            pl.len().alias("n"),
            (n == "").mean().alias("name_empty"),
            (a == "").mean().alias("addr_raw_empty"),
            (pl.col("addr_empty") == 1).mean().alias("addr_empty"),
            a.str.contains(NULLTOK).mean().alias("null_token"),
            ((pl.col("addr_empty") == 0) & (pl.col("nums") == "")).mean().alias("no_number"),
            ((pl.col("addr_empty") == 0) & (pl.col("state") == "")).mean().alias("no_state"),
            ((pl.col("addr_empty") == 0) & (pl.col("alpha_comps") == "")).mean().alias("no_alpha_comp"),
            a.str.contains(r"\b\d{5}\b").mean().alias("has_5dig"),
            a.str.contains(r"\b\d{6}\b").mean().alias("has_6dig"),
            (a.str.count_matches(",") + 1).mean().alias("n_comps"),
            pl.col("business_name").str.contains(r"[^\x00-\x7F]").mean().alias("name_nonascii"),
            (pl.col("legal") != "").mean().alias("has_legal"),
        ).with_columns(pl.lit(sp).alias("split"), pl.lit(s).alias("src")).collect()
        rows.append(r)
out = pl.concat(rows).sort("country", "split", "src")
cols = [c for c in out.columns if c not in ("split", "src", "country", "n")]
print(out.select("country", "split", "src", "n", *[pl.col(c).round(4) for c in cols]))
out.write_parquet(W + "missprof_v2.parquet")
