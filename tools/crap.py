"""Per-function CRAP scores: complexity^2 * (1 - coverage)^3 + complexity.

Coverage per function = covered (lines + branch arcs) / all (lines + branch arcs) in it.

Usage: .venv/bin/python -m pytest --cov=podedit --cov-branch --cov-report=json:coverage.json
       .venv/bin/python tools/crap.py coverage.json [threshold]
"""
import json
import sys
from pathlib import Path

import lizard


def main(cov_path: str, threshold: float = 12.0) -> int:
    files = json.loads(Path(cov_path).read_text())["files"]
    rows = []
    for fname, data in files.items():
        executed, missing = set(data["executed_lines"]), set(data["missing_lines"])
        hit_arcs = [a for a, _ in data.get("executed_branches", [])]
        miss_arcs = [a for a, _ in data.get("missing_branches", [])]
        for fn in lizard.analyze_file(fname).function_list:
            lines = set(range(fn.start_line, fn.end_line + 1))
            run = len(lines & executed) + sum(a in lines for a in hit_arcs)
            miss = len(lines & missing) + sum(a in lines for a in miss_arcs)
            cov = run / (run + miss) if run + miss else 1.0
            cc = fn.cyclomatic_complexity
            rows.append((cc * cc * (1 - cov) ** 3 + cc, cc, cov, f"{fname}:{fn.start_line} {fn.name}"))
    rows.sort(reverse=True)
    for crap, cc, cov, name in rows:
        flag = "  OVER" if crap > threshold else ""
        print(f"{crap:6.1f}  cc={cc:<3d} cov={cov:4.0%}  {name}{flag}")
    over = [r for r in rows if r[0] > threshold]
    print(f"\n{len(rows)} functions, max CRAP {rows[0][0]:.1f}, {len(over)} over {threshold}")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 12.0))
