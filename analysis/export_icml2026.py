from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


def _iter_table_files(source_root: Path, run_id: str) -> Iterable[Path]:
    root = source_root / "artifacts" / "tables"
    if not root.exists():
        return []
    for p in sorted(root.glob(f"{run_id}__*.md")):
        yield p
    for p in sorted(root.glob(f"{run_id}__*.json")):
        yield p


def _iter_demo_files(source_root: Path, run_id: str) -> Iterable[Path]:
    root = source_root / "artifacts" / "demos"
    if not root.exists():
        return []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        # Keep only the run-scoped demo folders.
        if f"/{run_id}/" in p.as_posix():
            yield p


def _iter_figure_files(source_root: Path, run_id: str) -> Iterable[Path]:
    root = source_root / "artifacts" / "figures"
    if not root.exists():
        return []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        # Keep only the run-scoped figure folders.
        if f"/{run_id}/" in p.as_posix():
            yield p


def _rel_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = Path(shutil.os.path.relpath(src, start=dst.parent))
    dst.symlink_to(rel)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def export_runs(
    run_ids: List[str],
    *,
    source_root: Path,
    out_root: Path,
    mode: str,
    include_figures: bool,
) -> Tuple[Path, List[Path]]:
    tables_root = out_root / "tables"
    demos_root = out_root / "demos"
    figures_root = out_root / "figures"
    tables_root.mkdir(parents=True, exist_ok=True)
    demos_root.mkdir(parents=True, exist_ok=True)
    if include_figures:
        figures_root.mkdir(parents=True, exist_ok=True)

    exported: List[Path] = []

    for rid in run_ids:
        for p in _iter_table_files(source_root, rid):
            dst = tables_root / p.name
            if mode == "copy":
                _copy_file(p, dst)
            else:
                _rel_symlink(p, dst)
            exported.append(dst)

        for p in _iter_demo_files(source_root, rid):
            # Preserve subpath under artifacts/demos
            rel = p.relative_to(source_root / "artifacts" / "demos")
            dst = demos_root / rel
            if mode == "copy":
                _copy_file(p, dst)
            else:
                _rel_symlink(p, dst)
            exported.append(dst)

        if include_figures:
            for p in _iter_figure_files(source_root, rid):
                rel = p.relative_to(source_root / "artifacts" / "figures")
                dst = figures_root / rel
                if mode == "copy":
                    _copy_file(p, dst)
                else:
                    _rel_symlink(p, dst)
                exported.append(dst)

    return out_root, exported


def write_index(
    out_root: Path,
    run_ids: List[str],
    exported: List[Path],
    *,
    mode: str,
    source_root: Path,
    source_root_display: str,
) -> Path:
    idx = out_root / "INDEX.md"
    paper_root_guess = out_root.parent.parent
    lines = [
        "# ICML artifact export index",
        "",
        f"- mode: `{mode}`",
        f"- source_root: `{source_root_display}`",
        "- source: `<source_root>/artifacts/*` (append-only)",
        "",
        "## Runs included",
        "",
    ]
    for rid in run_ids:
        lines.append(f"- `{rid}`")
    lines.append("")
    lines.append("## Exported paths")
    lines.append("")
    for p in exported:
        try:
            rel = p.relative_to(paper_root_guess).as_posix()
        except ValueError:
            rel = p.as_posix()
        lines.append(f"- `{rel}`")
    lines.append("")
    idx.write_text("\n".join(lines), encoding="utf-8")
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description="Export selected run artifacts into icml2026/ for paper integration.")
    ap.add_argument("--run", action="append", required=True, help="Run id (directory name under runs/). Repeatable.")
    ap.add_argument(
        "--source-root",
        default=".",
        help="Workspace root containing artifacts/ (default: current working directory).",
    )
    ap.add_argument(
        "--out-root",
        default="icml2026/artifacts",
        help="Output directory for exported artifacts (default: icml2026/artifacts).",
    )
    ap.add_argument(
        "--include-figures",
        action="store_true",
        help="Also export artifacts/figures/**/<run_id>/** into <out_root>/figures/.",
    )
    ap.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="symlink (default) keeps icml2026 small; copy makes a standalone folder.",
    )
    args = ap.parse_args()

    source_root_display = str(args.source_root)
    source_root = Path(args.source_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    out_root, exported = export_runs(
        args.run,
        source_root=source_root,
        out_root=out_root,
        mode=args.mode,
        include_figures=args.include_figures,
    )
    idx = write_index(
        out_root,
        args.run,
        exported,
        mode=args.mode,
        source_root=source_root,
        source_root_display=source_root_display,
    )
    print(idx.as_posix())


if __name__ == "__main__":
    main()
