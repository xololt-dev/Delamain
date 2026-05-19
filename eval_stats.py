import re
import ast
import sys
import statistics


def parse_file(filepath):
    runs = []
    with open(filepath) as f:
        content = f.read()

    blocks = re.split(r"Track seed:\s*\d+\n", content)
    for block in blocks[1:]:
        run = {}
        m = re.search(r"reward\s+([-\d.e]+)", block)
        if m:
            run["reward"] = float(m.group(1))

        m = re.search(r"episode length\s+(\d+)", block)
        if m:
            run["episode_length"] = int(m.group(1))

        m = re.search(r"info\s+(\{.*?\})", block)
        if m:
            try:
                run["info"] = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                run["info"] = {}

        m = re.search(r"actions_in_row:\s*([\d.e]+)", block)
        if m:
            run["actions_in_row"] = float(m.group(1))

        m = re.search(r"Fuel efficiency:\s*([\d.e]+)", block)
        if m:
            run["fuel_efficiency"] = float(m.group(1))

        if run:
            runs.append(run)

    return runs


def categorize(runs):
    success = fail = timeout = 0
    for r in runs:
        info = r.get("info", {})
        if "lap_finished" in info:
            if info["lap_finished"]:
                success += 1
            else:
                fail += 1
        else:
            timeout += 1
    return success, fail, timeout


def pct(values, p):
    """p-th percentile (0-100) with linear interpolation."""
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = k - f
    if f + 1 < len(s):
        return s[f] + c * (s[f + 1] - s[f])
    return s[f]


def fmt_stats(name, values):
    return (
        f"  {name}: min={min(values):.3f}  max={max(values):.3f}  "
        f"avg={statistics.mean(values):.3f}  median={statistics.median(values):.3f}  "
        f"p25={pct(values, 25):.3f}  p75={pct(values, 75):.3f}"
    )


def analyze_file(filepath):
    runs = parse_file(filepath)
    if not runs:
        print(f"No runs found in {filepath}")
        return

    rewards = [r["reward"] for r in runs]
    lengths = [r.get("episode_length") for r in runs if r.get("episode_length") is not None]
    actions_in_row = [r["actions_in_row"] for r in runs if r.get("actions_in_row") is not None]
    fuel = [r["fuel_efficiency"] for r in runs if r.get("fuel_efficiency") is not None]
    success, fail, timeout = categorize(runs)

    print(f"\n=== {filepath} ===")
    print(fmt_stats("reward", rewards))
    if lengths:
        print(fmt_stats("episode_length", lengths))
    if actions_in_row:
        print(fmt_stats("actions_in_row", actions_in_row))
    if fuel:
        print(fmt_stats("fuel_efficiency", fuel))
    print(f"  Success/Fail/Timeout: {success}/{fail}/{timeout}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_stats.py file1.txt [file2.txt ...]")
        sys.exit(1)

    for fp in sys.argv[1:]:
        try:
            analyze_file(fp)
        except FileNotFoundError:
            print(f"\nFile not found: {fp}")


if __name__ == "__main__":
    main()
