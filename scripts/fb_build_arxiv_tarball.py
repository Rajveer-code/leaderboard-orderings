"""Build the arXiv source tarball for manuscript/tmlr/arxiv.tex and prove it compiles from a clean folder.

arXiv runs pdflatex without bibtex, so the tarball ships arxiv.bbl. The file list is taken from the
latexmk .fls record (every file the build actually read inside the manuscript folder), so nothing is
guessed. The clean-room check unpacks the tarball into an empty temp folder and runs pdflatex twice.

Usage: python scripts/fb_build_arxiv_tarball.py   (run after `latexmk -pdf arxiv.tex`)
"""
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# ---- PATHS ----
REPO = Path(__file__).resolve().parents[1]
MS = REPO / "manuscript" / "tmlr"
MAIN = "arxiv"
OUT = REPO.parent / "flipbudget_arxiv_source.tar.gz"
SKIP_SUFFIXES = {".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".pdf_tex"}


def build_inputs():
    """Relative paths of every file the build read from inside the manuscript folder."""
    fls = (MS / f"{MAIN}.fls").read_text(encoding="utf-8", errors="replace").splitlines()
    files = set()
    for line in fls:
        if not line.startswith("INPUT "):
            continue
        p = Path(line[6:].strip())
        p = p if p.is_absolute() else (MS / p)
        try:
            rel = p.resolve().relative_to(MS.resolve())
        except ValueError:
            continue  # TeX distribution file, arXiv has its own
        if rel.suffix in SKIP_SUFFIXES or rel.name == f"{MAIN}.pdf":
            continue
        files.add(rel.as_posix())
    files.add(f"{MAIN}.bbl")
    return sorted(files)


def main():
    files = build_inputs()
    if f"{MAIN}.tex" not in files:
        sys.exit("arxiv.tex missing from the .fls record; run latexmk first")
    with tarfile.open(OUT, "w:gz") as tar:
        for rel in files:
            tar.add(MS / rel, arcname=rel)
    print(f"{OUT} : {len(files)} files, {OUT.stat().st_size / 1e6:.2f} MB")

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(OUT) as tar:
            tar.extractall(tmp, filter="data")
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{MAIN}.tex"],
                               cwd=tmp, capture_output=True, text=True, errors="replace")
        log = (Path(tmp) / f"{MAIN}.log").read_text(errors="replace")
        undefined = log.count("undefined")
        import fitz  # PyMuPDF
        pages = fitz.open(Path(tmp) / f"{MAIN}.pdf").page_count
        ref_pages = fitz.open(MS / f"{MAIN}.pdf").page_count
        print(f"clean-room pdflatex exit {r.returncode}; undefined {undefined}; "
              f"pages {pages} (repo build {ref_pages})")
        if r.returncode or undefined or pages != ref_pages:
            sys.exit(1)
        shutil.copy(Path(tmp) / f"{MAIN}.pdf", REPO.parent / "flipbudget_arxiv_cleanroom.pdf")


if __name__ == "__main__":
    main()
