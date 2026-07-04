import argparse
import pandas as pd
import sys

def main():
    p = argparse.ArgumentParser(description="Keep only rows with style == 'E' in a CSV.")
    p.add_argument("input", help="input CSV path")
    p.add_argument("output", nargs="?", help="output CSV path (defaults to overwrite input)")
    args = p.parse_args()

    out_path = args.output if args.output else args.input

    try:
        df = pd.read_csv(args.input, dtype=str)
    except Exception as e:
        print(f"Failed to read {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    if "style" not in df.columns:
        print("No 'style' column found in CSV.", file=sys.stderr)
        sys.exit(1)

    filtered = df[df["style"] == "E"].copy()
    filtered.to_csv(out_path, index=False)
    print(f"Wrote {len(filtered)} rows to {out_path}")

if __name__ == "__main__":
    main()