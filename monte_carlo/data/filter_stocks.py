import argparse
from pathlib import Path
import pandas as pd
import sys

def main():
    p = argparse.ArgumentParser(description="Keep only stock rows whose symbol appears as 'underlying' in options CSV.")
    p.add_argument("options_csv", nargs="?", default=str(Path(__file__).resolve().parents[1] / "data" / "2013-01-02options.csv"))
    p.add_argument("stocks_csv", nargs="?", default=str(Path(__file__).resolve().parents[1] / "data" / "2013-01-02stocks.csv"))
    p.add_argument("-o", "--output", help="output CSV path (defaults to filtered_<stocks_csv>)")
    args = p.parse_args()

    opt_path = Path(args.options_csv)
    stk_path = Path(args.stocks_csv)
    out_path = Path(args.output) if args.output else stk_path.with_name(f"filtered_{stk_path.name}")

    if not opt_path.exists():
        print(f"Options file not found: {opt_path}", file=sys.stderr); sys.exit(1)
    if not stk_path.exists():
        print(f"Stocks file not found: {stk_path}", file=sys.stderr); sys.exit(1)

    opts = pd.read_csv(opt_path, dtype=str)
    if "underlying" not in opts.columns:
        print("No 'underlying' column in options CSV.", file=sys.stderr); sys.exit(1)
    underlyings = set(opts["underlying"].dropna().astype(str).str.strip())

    stocks = pd.read_csv(stk_path, dtype=str)
    if "symbol" not in stocks.columns:
        print("No 'symbol' column in stocks CSV.", file=sys.stderr); sys.exit(1)

    filtered = stocks[stocks["symbol"].astype(str).str.strip().isin(underlyings)].copy()
    filtered.to_csv(out_path, index=False)
    print(f"Wrote {len(filtered)} rows to {out_path}")

if __name__ == "__main__":
    main()