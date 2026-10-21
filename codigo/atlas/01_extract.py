"""Paso 1: extrae texto, DOI y metadatos de cada PDF en data/raw -> data/extracted.json"""
import hashlib
import json
import os
import re
import sys
import unicodedata

import fitz  # PyMuPDF

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "extracted.json")

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>,;]+)", re.I)
PMID_RE = re.compile(r"PMID:?\s*(\d{6,9})")
fitz.TOOLS.mupdf_display_errors(False)


def clean_doi(d):
    d = d.rstrip(").].,;:'")
    # cortar sufijos típicos pegados al DOI en el texto extraído
    d = re.split(r"(?i)(?:received|accepted|published|copyright|citation|https?:)", d)[0]
    return d.rstrip(").].,;:'/").lower()


def extract_abstract(text):
    m = re.search(r"(?is)\b(abstract|summary|resumen)\b[:.\s]*(.{200,2500}?)"
                  r"(?=\n\s*(?:keywords?|key words|introduction|background\n|1\.?\s+introduction|©|\bintroduction\b))",
                  text)
    if m:
        return re.sub(r"\s+", " ", m.group(2)).strip()
    return ""


def main():
    rows = []
    files = []
    for root, _, fs in os.walk(RAW):
        for f in fs:
            if f.lower().endswith(".pdf"):
                files.append(os.path.join(root, f))
    files.sort()
    for i, path in enumerate(files):
        rel = os.path.relpath(path, RAW).replace("\\", "/")
        with open(path, "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        row = {"file": rel, "md5": md5, "filename_title": os.path.splitext(os.path.basename(path))[0]}
        try:
            doc = fitz.open(path)
            row["pages"] = doc.page_count
            row["meta_title"] = (doc.metadata or {}).get("title", "") or ""
            text = "".join(doc[p].get_text() for p in range(min(3, doc.page_count)))
            doc.close()
            text = unicodedata.normalize("NFKC", text)
            text = re.sub(r"[\u200b-\u200f\u2060\ufeff\u00ad]", "", text)
        except Exception as e:
            print("ERR", rel, e, file=sys.stderr)
            text = ""
        row["text"] = text[:12000]
        dois = list(dict.fromkeys(clean_doi(d) for d in DOI_RE.findall(text[:8000])))
        row["dois"] = dois
        row["doi"] = dois[0] if dois else ""
        pm = PMID_RE.search(text[:6000])
        row["pmid"] = pm.group(1) if pm else ""
        row["pdf_abstract"] = extract_abstract(text)
        rows.append(row)
        if i % 50 == 0:
            print(f"{i}/{len(files)}")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    n_doi = sum(1 for r in rows if r["doi"])
    n_abs = sum(1 for r in rows if r["pdf_abstract"])
    n_txt = sum(1 for r in rows if len(r["text"]) > 500)
    print(f"{len(rows)} PDFs | con DOI: {n_doi} | abstract en PDF: {n_abs} | con texto: {n_txt}")
    print("duplicados (md5):", len(rows) - len({r['md5'] for r in rows}))


if __name__ == "__main__":
    main()
