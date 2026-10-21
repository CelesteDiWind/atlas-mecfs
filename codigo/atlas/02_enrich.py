"""Paso 2: enriquece cada PDF con metadatos de OpenAlex (por DOI) y Crossref (búsqueda por título)
-> data/papers.json (deduplicado). Las respuestas se cachean en data/cache/."""
import difflib
import hashlib
import json
import os
import re
import time

import requests

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "cache")
os.makedirs(CACHE, exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "mecfs-bibliography-map/0.1"

STOP = set("the of and in with for a an to on by from at as is are among patients study".split())
DOMAIN = set("myalgic encephalomyelitis chronic fatigue syndrome mecfs covid long post".split())


def cached_get(url, params=None):
    key = hashlib.md5((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    for attempt in range(4):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 404:
                data = None
                break
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            break
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    else:
        return None  # sin cachear: se reintentará en la próxima ejecución
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    time.sleep(0.12)
    return data


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())


def title_in_text(title, text):
    """Fracción de palabras significativas del título que aparecen en el texto del PDF."""
    words = [w for w in norm(title).split() if len(w) > 3 and w not in STOP]
    if not words:
        return 0.0
    t = set(norm(text).split())
    return sum(w in t for w in words) / len(words)


def similar(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def openalex_by_doi(doi):
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return cached_get(f"https://api.openalex.org/works/doi:{doi}")


def crossref_search(q):
    d = cached_get("https://api.crossref.org/works",
                   {"query.bibliographic": q[:300], "rows": 5, "select": "DOI,title,score"})
    return (d or {}).get("message", {}).get("items", []) if d else []


def inverted_to_text(inv):
    if not inv:
        return ""
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))


def good_query_title(r):
    mt = r.get("meta_title", "").strip()
    if len(mt) > 25 and not re.search(r"(?i)microsoft word|untitled|\.docx?|\.pdf|^doi|^pii|^s\d{4}", mt):
        return mt
    return r["filename_title"]


def sig_words(s):
    return [w for w in norm(s).split() if len(w) > 3 and w not in STOP]


def filename_cover(title, fname):
    """Fracción de palabras del nombre de archivo presentes en el título candidato (None si no es descriptivo)."""
    fw = sig_words(re.sub(r"[_\-]", " ", fname))
    if len(fw) < 4:
        return None
    tw = set(norm(title).split())
    return sum(w in tw for w in fw) / len(fw)


def phrase_in_text(phrase, text):
    """Mejor similitud entre `phrase` y una ventana contigua del texto (0..1)."""
    p = " ".join(norm(phrase).split())
    t = " ".join(norm(text).split())
    L = len(p)
    if not L or not t:
        return 0.0
    sm = difflib.SequenceMatcher(None, "", p, autojunk=False)
    best = 0.0
    for i in range(0, max(1, len(t) - L + 1), 3):
        sm.set_seq1(t[i:i + L])
        if sm.real_quick_ratio() > best and sm.quick_ratio() > best:
            best = max(best, sm.ratio())
    return best


def score(title, r):
    """Puntaje de que `title` sea el título del PDF r (texto de las 1as páginas + nombre de archivo)."""
    if len(sig_words(title)) < 3:
        return 0.0
    text = r["text"][:4000]
    st = phrase_in_text(title, text) if len(text) > 500 else similar(title, good_query_title(r))
    fc = filename_cover(title, r["filename_title"])
    if fc is not None and fc < 0.3:
        fname = re.sub(r"[_\-]", " ", r["filename_title"])
        if phrase_in_text(fname, text) >= 0.8:
            return 0.0  # el PDF contiene el paper del nombre de archivo, y este candidato es otro
        if fc < 0.2 and st < 0.95:
            return 0.0  # el nombre de archivo describe otro paper
    if len([w for w in sig_words(title) if w not in DOMAIN]) < 2 and (fc or 0) < 0.5:
        return 0.0  # título genérico ("Myalgic Encephalomyelitis/Chronic Fatigue Syndrome")
    if len(sig_words(title)) < 5 and st < 0.95:
        return 0.0  # título corto/genérico: exigir coincidencia total
    return st + 0.3 * (fc or 0)


def match(r):
    """Devuelve (obra OpenAlex, método) o (None, None): el candidato con mejor puntaje."""
    best = (0.0, None, None)
    # 1) DOIs encontrados en el PDF (bonus si el DOI está en la cabecera de la 1a página)
    head = r["text"][:800].lower()
    for doi in r.get("dois", [])[:6]:
        w = openalex_by_doi(doi)
        if w and w.get("title"):
            sc = score(w["title"], r)
            if sc and doi in head:
                sc += 0.25
            if sc > best[0]:
                best = (sc, w, "doi")
    if best[0] >= 0.85:
        return best[1], best[2]
    # 2) búsqueda en Crossref por título y, si no, por el inicio del texto -> OpenAlex por DOI
    q = good_query_title(r)
    queries = [q]
    if len(r["text"]) > 500:
        queries.append(" ".join(r["text"][:300].split()))
    for qq in queries:
        for it in crossref_search(qq):
            t = (it.get("title") or [""])[0]
            sc = max(score(t, r), 1.0 if similar(t, q) > 0.85 else 0.0)
            if sc > best[0] and sc >= 0.85:
                w = openalex_by_doi(it["DOI"])
                if w:
                    best = (sc, w, "title")
        if best[0] >= 0.85:
            break
    if best[0] >= 0.85:
        return best[1], best[2]
    return None, None


def main():
    with open(os.path.join(DATA, "extracted.json"), encoding="utf-8") as fh:
        rows = json.load(fh)

    by_key = {}
    stats = {"doi": 0, "title": 0, "none": 0}
    seen_md5 = {}
    for i, r in enumerate(rows):
        if r["md5"] in seen_md5:
            by_key[seen_md5[r["md5"]]]["files"].append(r["file"])
            continue
        w, how = match(r)
        stats[how or "none"] += 1
        if w:
            loc = w.get("primary_location") or {}
            src = loc.get("source") or {}
            oa = w.get("open_access") or {}
            paper = {
                "key": w["id"].rsplit("/", 1)[-1],
                "title": w.get("title") or r["filename_title"],
                "year": w.get("publication_year"),
                "authors": [a["author"]["display_name"] for a in w.get("authorships", [])[:12]],
                "n_authors": len(w.get("authorships", [])),
                "venue": src.get("display_name") or "",
                "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                "url": w.get("doi") or loc.get("landing_page_url") or "",
                "oa_url": oa.get("oa_url") or "",
                "is_oa": bool(oa.get("is_oa")),
                "cited_by": w.get("cited_by_count", 0),
                "type": w.get("type", ""),
                "topics": [t["display_name"] for t in (w.get("topics") or [])[:3]],
                "keywords": [k["display_name"] for k in (w.get("keywords") or [])[:6]],
                "abstract": inverted_to_text(w.get("abstract_inverted_index")),
                "abstract_source": "openalex",
                "matched_by": how,
            }
        else:
            paper = {
                "key": "local-" + r["md5"][:10],
                "title": good_query_title(r),
                "year": None, "authors": [], "n_authors": 0, "venue": "",
                "doi": r.get("doi", ""), "url": ("https://doi.org/" + r["doi"]) if r.get("doi") else "",
                "oa_url": "", "is_oa": False, "cited_by": 0, "type": "", "topics": [], "keywords": [],
                "abstract": "", "abstract_source": "", "matched_by": "none",
            }
        if not paper["abstract"] and r["pdf_abstract"]:
            paper["abstract"], paper["abstract_source"] = r["pdf_abstract"], "pdf"
        paper["files"] = [r["file"]]
        paper["_text"] = r["text"]
        if paper["key"] in by_key:  # mismo paper con otro PDF
            by_key[paper["key"]]["files"].append(r["file"])
        else:
            by_key[paper["key"]] = paper
        seen_md5[r["md5"]] = paper["key"]
        if i % 50 == 0:
            print(f"{i}/{len(rows)} {stats}", flush=True)

    papers = list(by_key.values())
    with open(os.path.join(DATA, "papers.json"), "w", encoding="utf-8") as fh:
        json.dump(papers, fh, ensure_ascii=False, indent=1)
    print("match:", stats)
    print("papers únicos:", len(papers),
          "| con abstract:", sum(1 for p in papers if p["abstract"]),
          "| abstract openalex:", sum(1 for p in papers if p["abstract_source"] == "openalex"),
          "| open access:", sum(1 for p in papers if p["oa_url"]))


if __name__ == "__main__":
    main()
