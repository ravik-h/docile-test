#!/usr/bin/env python3
"""
Split a flat directory of PDFs into subdirectories of at most N files (default 50).

Usage:
  python shard_pdfs.py docile/pdfs out/docile               # copy, 50 per folder
  python shard_pdfs.py docile/pdfs out/docile --move        # move instead of copy
  python shard_pdfs.py docile/pdfs out/docile --link        # move, then leave a link behind in the
                                                            # original folder so nothing is stored twice
  python shard_pdfs.py docile/pdfs out/docile --shard-size 25
  python shard_pdfs.py docile/pdfs out/docile --ids docile/val.json    # only the IDs listed
  python shard_pdfs.py docile/pdfs out/docile --annotations docile/annotations
  python shard_pdfs.py docile/pdfs out/docile --exclude-scanned   # skip image-only / OCR'd scans

Output:
  out/docile/0000/<file>.pdf ... 0001/ ...      <- PDFs only, nothing else in the tree
  out/docile_manifest.jsonl                     <- written beside the tree, not inside it
  (one line per file: path, original name, size, and document metadata when
  --annotations points at the DocILE annotations folder). --no-manifest skips it.

Files are sorted by name before sharding so the layout is reproducible.

Scan detection (needs `pip install pdfplumber`): a page counts as scanned when
it yields almost no extractable text, or when a single image covers most of the
page (an OCR'd scan: picture underneath, invisible text on top). A document is
"scanned" if its first page is; the verdict and the raw numbers go into the
manifest as `scanned`, `text_chars`, `image_coverage`. --exclude-scanned leaves
those documents where they are and lists them in <dst>_excluded.txt.

--link moves each PDF into its shard and replaces the original with a relative
symlink pointing at the new location, so tools that expect the original flat
folder (e.g. the docile library) keep working while each PDF exists only once.
On Windows, symlinks need Developer Mode or admin rights; if creating one fails
the script falls back to a hard link (same volume only) and says so. Re-running
with --link is safe: existing links are left alone.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def load_ids(path: Path) -> set[str] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):  # tolerate {"ids": [...]} style
        data = next(iter(data.values()))
    return {str(x) for x in data}


def read_metadata(annotations_dir: Path | None, docid: str) -> dict:
    """Pull the useful bits out of a DocILE annotation file, if present."""
    if annotations_dir is None:
        return {}
    ann_path = annotations_dir / f"{docid}.json"
    if not ann_path.exists():
        return {}
    try:
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"annotation_error": "invalid json"}
    meta = ann.get("metadata", {}) or {}
    fields = ann.get("field_extractions", []) or []
    line_items = ann.get("line_item_extractions", []) or []

    # Flatten header fields to fieldtype -> text (a list when a type occurs more than once,
    # e.g. two vendor_name boxes on one page).
    flat: dict[str, str | list[str]] = {}
    for f in fields:
        ft, txt = f.get("fieldtype"), (f.get("text") or "").strip()
        if not ft:
            continue
        if ft in flat:
            flat[ft] = (flat[ft] if isinstance(flat[ft], list) else [flat[ft]]) + [txt]
        else:
            flat[ft] = txt

    return {
        "document_type": meta.get("document_type"),
        "language": meta.get("language"),
        "currency": meta.get("currency"),
        "page_count": meta.get("page_count"),
        "cluster_id": meta.get("cluster_id"),
        "source": meta.get("source"),
        "original_filename": meta.get("original_filename"),
        "n_fields": len(fields),
        "n_line_items": len({li.get("line_item_id") for li in line_items}),
        "fields": flat,
    }


def scan_check(pdf_path: Path, min_text_chars: int, max_image_coverage: float) -> dict:
    """Look at page 1 and decide whether the PDF is a scan.

    Returns {"scanned": bool|None, "text_chars": int, "image_coverage": float, "scan_reason": str}.
    scanned is None if the file could not be opened.
    """
    try:
        import pdfplumber  # imported lazily so the script runs without it if the flag isn't used
    except ImportError:
        sys.exit("scan detection needs pdfplumber: pip install pdfplumber")
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if not pdf.pages:
                return {"scanned": None, "text_chars": 0, "image_coverage": 0.0, "scan_reason": "no pages"}
            page = pdf.pages[0]
            text_chars = len((page.extract_text() or "").strip())
            page_area = float(page.width) * float(page.height) or 1.0
            # largest single image as a fraction of the page
            coverage = 0.0
            for im in page.images:
                w = max(0.0, float(im["x1"]) - float(im["x0"]))
                h = max(0.0, float(im["bottom"]) - float(im["top"]))
                coverage = max(coverage, (w * h) / page_area)
            coverage = min(coverage, 1.0)
    except Exception as exc:
        return {"scanned": None, "text_chars": 0, "image_coverage": 0.0, "scan_reason": f"unreadable: {exc}"}

    if text_chars < min_text_chars:
        return {"scanned": True, "text_chars": text_chars, "image_coverage": round(coverage, 3),
                "scan_reason": "no text layer"}
    if coverage > max_image_coverage:
        return {"scanned": True, "text_chars": text_chars, "image_coverage": round(coverage, 3),
                "scan_reason": "full-page image with text layer (OCR'd scan)"}
    return {"scanned": False, "text_chars": text_chars, "image_coverage": round(coverage, 3), "scan_reason": ""}


def leave_link(original: Path, target: Path) -> bool:
    """Create original -> target as a relative symlink; fall back to a hard link.
    Returns True if the hard-link fallback was used."""
    rel = os.path.relpath(target, original.parent)
    try:
        original.symlink_to(rel)
        return False
    except (OSError, NotImplementedError):
        try:
            os.link(target, original)
            return True
        except OSError as exc:
            sys.exit(f"could not link {original} -> {target}: {exc}\n"
                     f"The file has already been moved to {target}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="directory containing the PDFs")
    ap.add_argument("dst", type=Path, help="output directory (created if missing)")
    ap.add_argument("--shard-size", type=int, default=50, help="max files per subfolder (default 50)")
    ap.add_argument("--move", action="store_true", help="move files instead of copying")
    ap.add_argument("--link", action="store_true",
                    help="move files, then leave a symlink at each original path pointing into the shards")
    ap.add_argument("--ids", type=Path, default=None, help="JSON list of doc IDs to include (e.g. train.json)")
    ap.add_argument("--annotations", type=Path, default=None, help="DocILE annotations dir, to enrich the manifest")
    ap.add_argument("--ext", default=".pdf", help="file extension to include (default .pdf)")
    ap.add_argument("--no-manifest", action="store_true", help="don't write a manifest at all")
    ap.add_argument("--exclude-scanned", action="store_true",
                    help="skip PDFs that look like scans (image-only or OCR'd image)")
    ap.add_argument("--min-text-chars", type=int, default=40,
                    help="fewer extractable chars than this on page 1 => scanned (default 40)")
    ap.add_argument("--max-image-coverage", type=float, default=0.7,
                    help="a page-1 image covering more than this fraction => scanned (default 0.7)")
    args = ap.parse_args()

    if not args.src.is_dir():
        sys.exit(f"not a directory: {args.src}")
    if args.shard_size < 1:
        sys.exit("--shard-size must be >= 1")

    wanted = load_ids(args.ids)
    files = sorted(p for p in args.src.iterdir() if p.is_file() and p.suffix.lower() == args.ext.lower())
    if wanted is not None:
        files = [p for p in files if p.stem in wanted]
        missing = wanted - {p.stem for p in files}
        if missing:
            print(f"warning: {len(missing)} IDs in {args.ids.name} have no {args.ext} in {args.src}", file=sys.stderr)
    if not files:
        sys.exit(f"no {args.ext} files found in {args.src}")

    scan_info: dict[Path, dict] = {}
    if args.exclude_scanned:
        print(f"checking {len(files)} PDFs for scans...", file=sys.stderr)
        kept, excluded = [], []
        for k, p in enumerate(files):
            info = scan_check(p.resolve() if p.is_symlink() else p, args.min_text_chars, args.max_image_coverage)
            scan_info[p] = info
            (excluded if info["scanned"] else kept).append(p)  # unreadable (None) is kept, flagged in manifest
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(files)} checked, {len(excluded)} scanned so far", file=sys.stderr)
        excl_path = args.dst.parent / f"{args.dst.name}_excluded.txt"
        excl_path.parent.mkdir(parents=True, exist_ok=True)
        excl_path.write_text("".join(f"{p.name}\t{scan_info[p]['scan_reason']}\n" for p in excluded), encoding="utf-8")
        print(f"excluded {len(excluded)} scanned PDFs (listed in {excl_path}); sharding {len(kept)}", file=sys.stderr)
        files = kept
        if not files:
            sys.exit("nothing left to shard after excluding scans")

    args.dst.mkdir(parents=True, exist_ok=True)
    transfer = shutil.move if (args.move or args.link) else shutil.copy2
    fell_back_to_hardlink = False
    n_shards = (len(files) + args.shard_size - 1) // args.shard_size
    width = max(4, len(str(n_shards - 1)))

    manifest_path = args.dst.parent / f"{args.dst.name}_manifest.jsonl"
    manifest = open(os.devnull if args.no_manifest else manifest_path, "w", encoding="utf-8")
    with manifest:
        for i, src in enumerate(files):
            shard = args.dst / f"{i // args.shard_size:0{width}d}"
            shard.mkdir(exist_ok=True)
            dst = shard / src.name
            if src.is_symlink():  # already sharded on a previous --link run
                dst = src.resolve()
                size = dst.stat().st_size
            else:
                size = src.stat().st_size
                transfer(str(src), str(dst))
                if args.link:
                    fell_back_to_hardlink |= leave_link(src, dst)
            record = {
                "path": dst.resolve().relative_to(args.dst.resolve()).as_posix(),
                "id": src.stem,
                "file": src.name,
                "bytes": size,
                **read_metadata(args.annotations, src.stem),
                **(scan_info.get(src) or {}),
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(files)}...", file=sys.stderr)

    verb = "moved" if (args.move or args.link) else "copied"
    if fell_back_to_hardlink:
        print("note: symlinks were not permitted; hard links were used instead (fine on the same volume)", file=sys.stderr)
    print(f"done: {len(files)} files {verb} into {n_shards} folders of <= {args.shard_size} -> {args.dst}")
    if not args.no_manifest:
        print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
