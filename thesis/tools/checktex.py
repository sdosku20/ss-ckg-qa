"""
Pre-flight check for the thesis sources. Run it before every LaTeX build.

    python thesis/tools/checktex.py

It catches the three mistakes that cost the most time in a LaTeX project,
without needing a LaTeX installation:

  1. a \\ref to a label that does not exist   -> renders as "??" in the PDF
  2. a \\cite key that is not in the .bib     -> renders as "[?]" in the PDF
  3. a chapter drifting off its page budget   -> found only at the end otherwise

Exit code is 1 if (1) or (2) found anything, so it can gate a build.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Words per typeset page for this template (memoir, a4paper, 10pt, oneside),
# calibrated once against a compiled PDF. Re-calibrate if the class options
# change: word_count_of_a_chapter / its_page_count_in_the_pdf.
WORDS_PER_PAGE = 500

# The plan from STATUS.md. Chapters not listed here are not budgeted yet.
PAGE_BUDGET = {
    "Chapters/Introduction.tex": 5,
    "Chapters/Background.tex": 8,
    "Chapters/GRetrieverPCST.tex": 6,
    "Chapters/Methodology.tex": 8,
    "Chapters/Implementation.tex": 6,
    "Chapters/Results.tex": 8,
    "Chapters/Conclusion.tex": 3,
}

BEGIN = re.compile(r"\\begin\{([^}]+)\}")
END = re.compile(r"\\end\{([^}]+)\}")
LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\(?:ref|autoref|eqref|nameref)\{([^}]+)\}")
CITE = re.compile(r"\\cite[a-zA-Z]*\{([^}]+)\}")
BIBKEY = re.compile(r"@\w+\{\s*([^,\s]+)\s*,")
# A comment is a % that is not escaped as \%. Stripped before counting words so
# the editorial notes do not inflate the page estimate.
COMMENT = re.compile(r"(?<!\\)%.*")


def sources(root: Path) -> dict[Path, str]:
    files = sorted(root.rglob("*.tex"))
    return {
        p: p.read_text(encoding="utf-8", errors="replace")
        for p in files
        if "Template" not in p.parts and not p.name.endswith(".bak")
    }


def check_structure(src: dict, root) -> list:
    """Catch the two syntax errors that a missing LaTeX install cannot.

    There is no compiler on every machine this project is edited from, so a
    stray brace or an unclosed environment would otherwise be discovered only at
    build time, possibly by a supervisor. Neither check is a substitute for
    compiling; both are cheap and catch the common cases.
    """
    problems = []
    for path, text in src.items():
        rel = path.relative_to(root).as_posix()
        body = COMMENT.sub("", text)

        # Environments must nest. Report the first mismatch with its name, since
        # an unclosed environment usually swallows the rest of the file.
        stack = []
        for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
            kind, name = match.group(1), match.group(2)
            if kind == "begin":
                stack.append(name)
            elif not stack:
                problems.append(f"{rel}: \\end{{{name}}} with no matching \\begin")
                break
            elif stack[-1] != name:
                problems.append(f"{rel}: \\end{{{name}}} closes \\begin{{{stack[-1]}}}")
                break
            else:
                stack.pop()
        if stack:
            problems.append(f"{rel}: unclosed environment(s) {stack}")

        # Braces, ignoring \{ and \} escapes.
        depth = 0
        for match in re.finditer(r"(?<!\\)[{}]", body):
            depth += 1 if match.group() == "{" else -1
            if depth < 0:
                problems.append(f"{rel}: unmatched closing brace")
                break
        if depth > 0:
            problems.append(f"{rel}: {depth} unclosed brace(s)")
    return problems


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    src = sources(root)
    if not src:
        print(f"no .tex files under {root}")
        return 1

    body = "\n".join(COMMENT.sub("", text) for text in src.values())
    labels = set(LABEL.findall(body))
    refs = set(REF.findall(body))
    cites = {k.strip() for group in CITE.findall(body) for k in group.split(",") if k.strip()}

    bib_path = root / "references.bib"
    bib = set(BIBKEY.findall(bib_path.read_text(encoding="utf-8", errors="replace")))

    dangling = sorted(refs - labels)
    missing = sorted(cites - bib)
    structural = check_structure(src, root)

    print(f"{len(src)} .tex files, {len(labels)} labels, {len(cites)} distinct citations")
    print(f"dangling \\ref  : {', '.join(dangling) if dangling else 'none'}")
    print(f"missing bib key: {', '.join(missing) if missing else 'none'}")
    print(f"structure      : {'; '.join(structural) if structural else 'balanced'}")

    unused = sorted(bib - cites)
    if unused:
        print(f"unused bib keys: {', '.join(unused)}")

    print("\npages (estimate at %d words/page):" % WORDS_PER_PAGE)
    total = 0.0
    for path, text in src.items():
        rel = path.relative_to(root).as_posix()
        words = len(COMMENT.sub("", text).split())
        if words < 120:
            continue
        pages = words / WORDS_PER_PAGE
        total += pages
        budget = PAGE_BUDGET.get(rel)
        if budget is None:
            note = ""
        elif pages > budget * 1.15:
            note = f"  OVER budget {budget}"
        elif pages < budget * 0.6:
            note = f"  under budget {budget}"
        else:
            note = f"  ok ({budget})"
        print(f"  {rel:34s} {words:6d} words  ~{pages:4.1f} p{note}")
    planned = sum(PAGE_BUDGET.values())
    print(f"  {'TOTAL (written)':34s} {'':6s}        ~{total:4.1f} p   of {planned} planned")

    return 1 if (dangling or missing or structural) else 0


if __name__ == "__main__":
    sys.exit(main())
