"""Indonesian prose style metrics for the draft (own implementation of the humanizer-indonesia metric set,
preset `akademik`). Prose only: headings, captions, tables, and the reference list are skipped.

Usage: uv run --with python-docx python paper/style_metrics.py paper/draft_paper_id.docx
"""
import re
import statistics
import sys

from docx import Document

CONNECTORS = ["selain itu", "namun demikian", "di sisi lain", "oleh karena itu", "dengan demikian", "lebih lanjut",
              "sementara itu", "akan tetapi", "oleh sebab itu", "walaupun demikian", "sayangnya", "di samping itu",
              "lebih jauh lagi", "sehubungan dengan itu"]
LIGHT_VERB = re.compile(r"\b(?:melakukan|memberikan|mengalami|melaksanakan|menjalankan|mengadakan)\s+(?:pe|ke)[a-z]{3,}\b", re.I)
FILLER = re.compile(r"\b(?:akan|telah|sudah|dapat|bisa|suatu|untuk|yang|melalui|hal|merupakan|adalah|dilakukan|"
                    r"terhadap|tersebut|ini|itu)\b", re.I)
COORD = re.compile(r",\s*(?:dan|serta|lalu|kemudian|sehingga|sedangkan|tetapi)\b", re.I)
DENIAL = re.compile(r"\bbukan\s+berarti\b|\bbukan\s+sekadar\b[^.]{0,80}?\bmelainkan\b|\bbukan\s+[^.,]{1,60},\s*melainkan\b|,\s*bukan\s+[^.,]{1,60}\.", re.I)
MONO_OPENERS = re.compile(r"(?:^|\. )(?:Konsekuensinya|Sedangkan|Pertanyaannya sederhana|Hasilnya tegas|Perhatikan bahwa|Mari kita)", re.I)
# preset akademik: (min, max)
TARGET = {"mean_len": (14.0, 20.0), "sd_len": (6.4, None), "long_pct": (None, 20.0), "coord": (None, 1.50),
          "connectors": (None, 0.40), "light_verbs": (None, 0.40), "enum_seperti": (None, 1.00),
          "merupakan": (None, 4.00), "dashes": (None, 0)}


def prose(path: str) -> str:
    keep = []
    for p in Document(path).paragraphs:
        t = p.text.strip()
        if not t or p.style.name.startswith(("Heading", "Title")):
            continue
        if all(r.italic for r in p.runs if r.text.strip()):  # English abstract and other fully italic notes
            continue
        if t.startswith(("Daftar Pustaka", "[")) or re.match(r"^(Tabel|Gambar) \d+\.", t):
            continue
        keep.append(t)
    text = "\n".join(keep)
    start = text.find("Model fondasi deret waktu")  # skip title block
    return text[start:] if start > 0 else text


def sentences(text: str) -> list[str]:
    t = re.sub(r"\b(dkk|dll|hlm|No)\.", r"\1<D>", text)
    t = re.sub(r"(\d)[.,](\d)", r"\1<C>\2", t)
    out = [s.replace("<D>", ".").replace("<C>", ",").strip() for s in re.split(r"(?<=[.!?])\s+", t.replace("\n", " "))]
    return [s for s in out if len(s.split()) > 3]


def measure(text: str) -> dict:
    sents = sentences(text)
    lens = [len(s.split()) for s in sents]
    words = sum(lens)
    low = text.lower()
    longs = [s for s in sents if len(s.split()) >= 28]
    seperti = re.findall(r"\b(?:seperti|antara lain|misalnya|di antaranya)\b[^.]{0,150}", low)
    return {
        "sentences": len(sents), "words": words,
        "mean_len": statistics.mean(lens), "sd_len": statistics.pstdev(lens),
        "short_pct": sum(l <= 10 for l in lens) / len(lens) * 100,
        "long_pct": len(longs) / len(sents) * 100,
        "coord": statistics.mean(len(COORD.findall(s)) for s in longs) if longs else 0.0,
        "connectors": sum(len(re.findall(r"\b" + c + r"\b", low)) for c in CONNECTORS) / words * 100,
        "light_verbs": len(LIGHT_VERB.findall(text)) / words * 100,
        "enum_seperti": sum(bool(re.search(r",\s*(?:dan|serta)\b", s)) for s in seperti) / words * 1000,
        "merupakan": len(re.findall(r"\bmerupakan\b", low)) / words * 1000,
        "dashes": text.count("—") + text.count("–"),
        "denials": len(DENIAL.findall(text)),
        "di_mana": len(re.findall(r"\b(?:dimana|di mana|yang mana)\b", low)),
        "mono_openers": len(MONO_OPENERS.findall(text)),
        "filler_per_1000": len(FILLER.findall(text)) / words * 1000,
        "examples_denial": [m.group(0) for m in DENIAL.finditer(text)][:6],
        "examples_dash": re.findall(r"[^ ]{0,15}[–—][^ ]{0,15}", text)[:12],
    }


def status(key: str, v: float) -> str:
    lo, hi = TARGET.get(key, (None, None))
    if lo is not None and v < lo:
        return "PERIKSA (rendah)"
    if hi is not None and v > hi:
        return "PERIKSA (tinggi)"
    return "LULUS" if key in TARGET else ""


def main(path: str) -> None:
    m = measure(prose(path))
    print(f"kalimat {m['sentences']}, kata {m['words']}")
    for k, v in m.items():
        if k in ("sentences", "words") or k.startswith("examples"):
            continue
        print(f"  {k:16s} {v:8.2f}  {status(k, v)}")
    print("  contoh penyangkalan:", m["examples_denial"])
    print("  contoh tanda pisah:", m["examples_dash"])


if __name__ == "__main__":
    main(sys.argv[1])
