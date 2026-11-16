"""Final stage: build README.md and its figures from the saved result files. No analysis is run here.

Every table, figure and number in the README is read from `results/` by `earnings_sentiment.readme`.

    python scripts/18_readme.py           write README.md and results/readme/figures/*.png
    python scripts/18_readme.py --check   exit 1 if README.md differs from what the saved results produce

Inputs   results/stage1..stage4 (CSV and JSON files only)
Outputs  README.md, results/readme/figures/headline.png, horizon_profile.png, quintiles.png
"""
from __future__ import annotations

import difflib
import sys

from earnings_sentiment import readme


def main(argv: list[str]) -> int:
    d = readme.load()
    text = readme.build_readme(d)
    if "--check" in argv:
        current = readme.README.read_text(encoding="utf-8") if readme.README.exists() else ""
        if current == text:
            print("README.md matches the saved results.")
            return 0
        sys.stdout.writelines(difflib.unified_diff(current.splitlines(True), text.splitlines(True),
                                                   "README.md (committed)", "README.md (from results)"))
        print("\nREADME.md differs from the saved results. Run scripts/18_readme.py to rebuild it.")
        return 1
    readme.make_figures(d)
    readme.README.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {readme.README} and figures in {readme.FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
