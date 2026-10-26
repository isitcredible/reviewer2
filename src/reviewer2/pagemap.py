"""Deterministic physical->printed page-map for Reviewer 2.

Replaces the LLM-guessed free-text PAGE_STRUCTURE range string with a real
per-physical-page printed-label map derived from the PDF itself, so every
page-aware stage (and the page checker) can cite the PRINTED page and a
deterministic guard can reject impossible page citations.

Sources, in priority order:
  1. Embedded /PageLabels number tree (authoritative when present; pypdf
     decodes the prefix cleanly).
  2. A top+bottom text scan of each page (fallback for PDFs with no
     /PageLabels), trusted only when dense + monotone.

The scan-vs-tree cross-check DRIVES the confidence: a substantial scan that
contradicts the embedded tree rejects the (malformed) tree; when neither source
is trustworthy the map degrades to identity + a "low" flag and the renderer
tells the model to cite the PDF index and say so. Never a confident wrong map.

Pure pypdf / PyMuPDF (both already pinned). No network. No Gemini.
"""
import re

# ---------------------------------------------------------------- numeral styles
def _to_roman(n, upper=False):
    if n <= 0:
        return str(n)
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"),
            (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"),
            (5, "v"), (4, "iv"), (1, "i")]
    out = []
    for v, s in vals:
        while n >= v:
            out.append(s)
            n -= v
    r = "".join(out)
    return r.upper() if upper else r

def _to_alpha(n, upper=False):
    if n <= 0:
        return str(n)
    s = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        s = chr(ord("a") + rem) + s
    return s.upper() if upper else s

def _fmt(num, style):
    if style == "/r":
        return _to_roman(num, upper=False)
    if style == "/R":
        return _to_roman(num, upper=True)
    if style == "/a":
        return _to_alpha(num, upper=False)
    if style == "/A":
        return _to_alpha(num, upper=True)
    return str(num)  # "/D" decimal, or none

# ---------------------------------------------------------------- /PageLabels
def labels_from_pagelabels(pdf_path, n_pages):
    """Printed labels (len n_pages) from the embedded /PageLabels tree, or None."""
    from pypdf import PdfReader
    r = PdfReader(pdf_path)
    root = r.trailer["/Root"]
    if "/PageLabels" not in root:
        return None
    nums = root["/PageLabels"].get("/Nums")
    if not nums:
        return None
    rules = []
    for i in range(0, len(nums) - 1, 2):
        spec = nums[i + 1]
        style = str(spec.get("/S")) if spec.get("/S") is not None else "/D"
        prefix = spec.get("/P", "")
        rules.append({"start": int(nums[i]), "style": style,
                      "prefix": str(prefix) if prefix else "", "st": int(spec.get("/St", 1))})
    rules.sort(key=lambda x: x["start"])
    if not rules:
        return None
    labels = []
    for idx in range(n_pages):
        rule = rules[0]
        for cand in rules:
            if cand["start"] <= idx:
                rule = cand
            else:
                break
        labels.append(rule["prefix"] + _fmt(rule["st"] + (idx - rule["start"]), rule["style"]))
    return labels

# ---------------------------------------------------------------- text scan
_LABEL_RE = re.compile(
    r"^\s*(?P<lab>(?:[A-Za-z]-?\d{1,4})|(?:\d{1,4})|(?:[ivxlcdm]{1,7})|(?:[IVXLCDM]{1,7}))\s*$")
_WATERMARK = re.compile(r"(downloaded|wiley|jstor|terms and conditions|https?://|doi:|all rights)", re.I)

def _scan_one(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    for l in lines[:2] + lines[-2:]:
        if _WATERMARK.search(l):
            continue
        m = _LABEL_RE.match(l)
        if m:
            return m.group("lab")
    return None

def _page_texts(pdf_path):
    """Plain text of every page. Uses PyMuPDF when installed (the optional
    ``pdf`` extra), else pypdf."""
    try:
        import fitz
        d = fitz.open(pdf_path)
        return [d[i].get_text("text") for i in range(d.page_count)]
    except ImportError:
        from pypdf import PdfReader
        return [(p.extract_text() or "") for p in PdfReader(pdf_path).pages]


def labels_from_text_scan(pdf_path, n_pages, texts=None):
    texts = texts if texts is not None else _page_texts(pdf_path)
    raw = [_scan_one(texts[i]) for i in range(n_pages)]
    recovered = sum(1 for x in raw if x is not None)
    return raw, (recovered / n_pages if n_pages else 0.0)

def _decimal_monotone_fraction(labels):
    ints = [(i, int(x)) for i, x in enumerate(labels)
            if x is not None and re.fullmatch(r"\d{1,4}", x)]
    if len(ints) < 2:
        return 0.0, len(ints)
    good = sum(1 for (a, va), (b, vb) in zip(ints, ints[1:]) if vb == va + 1)
    return good / (len(ints) - 1), len(ints)

def _norm(x):
    return x.strip().lstrip("0") if x else x

def _forward_fill(labels):
    out = list(labels)
    for i in range(len(out)):
        if out[i] is not None:
            continue
        j = i - 1
        while j >= 0 and (out[j] is None or not re.fullmatch(r"\d{1,4}", out[j] or "")):
            j -= 1
        out[i] = str(int(out[j]) + (i - j)) if j >= 0 else None
    return out

_MIN_CONTENT = 250  # chars of extractable text below which a page is treated as
                    # cover/divider/blank (never carries a citable printed folio)

def _roman_override(labels, scan):
    """Where the text scan read a clean ROMAN folio but the (tree-derived) label
    is arabic, trust the printed roman. A style mismatch is a strong, specific
    signal the tree mis-segmented the front matter (e.g. 'xi' labelled '13').
    Scan rarely fabricates a clean roman numeral, and this never fires on
    all-arabic papers."""
    out = list(labels)
    for i in range(len(out)):
        s = scan[i]
        if (s and re.fullmatch(r"[ivxlcdm]{1,7}", s)
                and out[i] and re.fullmatch(r"\d{1,4}", out[i])):
            out[i] = s
    return out

def _suppress_noncontent(labels, texts, scan):
    """Stop asserting a folio on pages that cannot carry one. Catches the
    zones where the mechanical map is most often confidently wrong: title/cover pages,
    blank dividers, supplement title pages, and tree labels extrapolated past
    the printed end onto a back cover. Conservative: only leading/trailing
    low-text runs, plus interior pages that are BOTH near-empty AND where the
    scan found no folio (a strong divider/blank signal)."""
    n = len(labels)
    txt = [len(texts[i].strip()) for i in range(n)]
    out = list(labels)
    i = 0
    while i < n and txt[i] < _MIN_CONTENT:   # leading covers / title / blanks
        out[i] = None
        i += 1
    j = n - 1
    while j >= 0 and txt[j] < _MIN_CONTENT:   # trailing back cover / blanks (off-the-end)
        out[j] = None
        j -= 1
    for k in range(n):                        # interior dividers / blanks
        if out[k] is not None and txt[k] < _MIN_CONTENT and scan[k] is None:
            out[k] = None
    return out

# ---------------------------------------------------------------- sections / ceilings
def _sections(labels):
    def split(lab):
        if not lab:
            return ("\x00unnumbered", "(unnumbered)")
        m = re.match(r"^([^\d]*)(\d+.*)$", lab)
        return (m.group(1), m.group(2)) if m else ("", lab)

    def num_of(lab):
        m = re.search(r"\d+", lab) if lab else None
        return int(m.group(0)) if m else None

    out, cur = [], None
    for idx, lab in enumerate(labels):
        pref = split(lab)[0]
        nv = num_of(lab)
        break_run = (cur is not None and pref == cur["prefix"] and nv is not None
                     and cur.get("last_num") is not None and nv != cur["last_num"] + 1)
        if cur is None or pref != cur["prefix"] or break_run:
            if cur:
                out.append(cur)
            cur = {"prefix": pref, "phys_start": idx, "phys_end": idx,
                   "printed_start": lab, "printed_end": lab, "last_num": nv}
        else:
            cur["phys_end"] = idx
            cur["printed_end"] = lab
            cur["last_num"] = nv if nv is not None else cur["last_num"]
    if cur:
        out.append(cur)
    return out

def _ceilings(labels):
    """Max numeric value per appendix-style prefix (for the citation guard).
    Only short alpha prefixes (A-, S-, A.) count; junk like 'Cover Page' is ignored."""
    ceils = {}
    for lab in labels:
        if not lab:
            continue
        m = re.match(r"^([^\d]*)(\d+)", lab)
        if not m:
            continue
        pref, num = m.group(1), int(m.group(2))
        if pref and re.fullmatch(r"[A-Za-z]{1,2}[-.]?", pref):
            ceils[pref] = max(ceils.get(pref, 0), num)
    return ceils

# ---------------------------------------------------------------- builder
def build_page_map(pdf_path):
    texts = _page_texts(pdf_path)
    n = len(texts)
    pl = labels_from_pagelabels(pdf_path, n)
    scan, frac = labels_from_text_scan(pdf_path, n, texts)
    mono, n_int = _decimal_monotone_fraction(scan)
    identity = [str(i + 1) for i in range(n)]

    def agreement(a_list):
        agree = total = 0
        for a, b in zip(a_list, scan):
            if b is None:
                continue
            total += 1
            agree += 1 if _norm(a) == _norm(b) else 0
        return agree, total

    if pl is not None:
        agree, total = agreement(pl)
        if total < 8 or (total and agree / total >= 0.60):
            method, confidence, labels = "pagelabels", "high", pl
            cross = f"{agree}/{total} scan-agree" if total else "no scan overlap"
        elif frac >= 0.70 and mono >= 0.90:
            method, confidence = "text-scan(pagelabels-rejected)", "high"
            labels = _forward_fill(scan)
            cross = f"/PageLabels REJECTED ({agree}/{total}); scan dense+monotone ({frac:.0%}/{mono:.0%})"
        else:
            method, confidence, labels = "conflict-degrade", "low", identity
            cross = f"/PageLabels contradicted ({agree}/{total}) and scan not clean ({frac:.0%}/{mono:.0%})"
    else:
        cross = f"recovered={frac:.0%}, decimal-monotone={mono:.0%} over {n_int} pages"
        if frac >= 0.70 and mono >= 0.90:
            method, confidence, labels = "text-scan", "high", _forward_fill(scan)
        else:
            method, confidence, labels = "identity-fallback", "low", identity

    # Hardening: only on a high-confidence map, so
    # the honest low-confidence fallback is untouched.
    if confidence == "high":
        if method == "pagelabels":
            labels = _roman_override(labels, scan)   # printed roman beats tree arabic
        labels = _suppress_noncontent(labels, texts, scan)  # no folio on covers/dividers/off-the-end

    return {"pdf": pdf_path, "n_pages": n, "method": method, "confidence": confidence,
            "cross_check": cross, "labels": labels, "scan": scan,
            "sections": _sections(labels), "ceilings": _ceilings(labels)}

# ---------------------------------------------------------------- renderer
def render_details(m):
    """The {{PAGE_STRUCTURE_DETAILS}} slot content. core.format_page_mapping_for_prompt
    delegates here. Returns the honest low-confidence message when no reliable map."""
    if not m or m.get("confidence") != "high":
        return ("Could not extract a reliable printed-page map from this PDF. Cite the "
                "PDF page index you see, and state that it is the PDF page index.")
    lines = []
    for s in m["sections"]:
        if s["prefix"] == "\x00unnumbered":
            span = (f"PDF page {s['phys_start']+1}" if s["phys_start"] == s["phys_end"]
                    else f"PDF pages {s['phys_start']+1}-{s['phys_end']+1}")
            lines.append(f"- {span}: unnumbered (cover/divider, do not cite)")
            continue
        name = "Main text" if not s["prefix"] else f"'{s['prefix']}' section"
        rng = (s["printed_start"] if s["printed_start"] == s["printed_end"]
               else f"{s['printed_start']}-{s['printed_end']}")
        # State the offset explicitly for plain-decimal sections so a stage cites
        # the printed folio without having to re-derive the shift each time (the
        # body off-by-one was the dominant failure: the model defaulted to the
        # PDF-viewer index whenever it had to do the subtraction itself).
        offset_note = ""
        if not s["prefix"] and s["printed_start"] and re.fullmatch(r"\d+", s["printed_start"]):
            off = (s["phys_start"] + 1) - int(s["printed_start"])
            if off == 0:
                offset_note = " (printed page = PDF viewer page)"
            elif off > 0:
                offset_note = f" (to cite a page here, SUBTRACT {off} from the PDF viewer page number)"
            else:
                offset_note = f" (to cite a page here, ADD {-off} to the PDF viewer page number)"
        lines.append(f"- {name}: PDF viewer pages {s['phys_start']+1}-{s['phys_end']+1} "
                     f"-> printed {rng}{offset_note}")
    if m["ceilings"]:
        ceil = "; ".join(f"{p}1 through {p}{c}" for p, c in m["ceilings"].items())
        lines.append(f"- The appendix/supplement runs {ceil}; a higher number in that "
                     f"section does not exist.")
    return "\n".join(lines)

# ---------------------------------------------------------------- Flash reconciliation
# Mechanical proposes, Flash (with vision) disposes, mechanical gates the result.
# The deterministic builder is reliable at arithmetic and blind to meaning: it
# cannot tell an unnumbered title/abstract page from printed page 1, cannot read a
# folio off a header or a scanned image, and trusts a malformed /PageLabels tree.
# Flash 2.5 reads the actual pages, so it can correct exactly those cases. But an
# LLM alone is what failed originally (the free-text page_structure guess), so its
# output is never trusted blind: every reconciled map must (a) cover all pages
# contiguously, (b) increment by exactly 1 within each numbered segment, and
# (c) agree with the INDEPENDENT text-scan wherever the scan read a real folio.
# A uniform off-by-one (the physical-vs-printed error) fails (c) and is rejected.

RECONCILE_PROMPT = """You are auditing a mechanically-derived page-number map for an academic PDF against the printed page numbers (folios) actually shown on the pages.

Definitions:
- The PHYSICAL (PDF viewer) page is a page's position in the file: the first page of the file is physical page 1.
- The PRINTED page is the folio actually printed on the page, usually at the foot (sometimes the head). It is what a reader cites.
They differ when there are unnumbered cover/title pages, roman-numeral front matter, or an appendix/supplement that restarts numbering (e.g. "A-1", "S-1").

A mechanical tool produced this draft map. It is often WRONG on title pages, front matter, and section boundaries (e.g. it may label an unnumbered title/abstract page as printed "1"):

{{MECHANICAL}}

This PDF has {{N}} physical pages.

YOUR JOB: look at the ACTUAL pages and output the corrected map as a list of contiguous numbering SEGMENTS that together cover physical pages 1 through {{N}} with NO gaps and NO overlaps. For each segment read the folio printed on its FIRST and its LAST physical page off the page image.

RULES:
- A page that shows NO printed folio (an unnumbered cover, title/abstract page, or a blank divider) belongs to an UNNUMBERED segment. NEVER invent a number for it, even if the mechanical map gave it one.
- Within one numbered segment the folio MUST increase by exactly 1 per physical page. When the style or prefix changes (arabic -> roman; body -> "A-1" appendix) or the count restarts, START A NEW SEGMENT there.
- Read the folios yourself and trust them over the mechanical map.

Output EXACTLY this and nothing else (no commentary, no markdown):

RECONCILED PAGE MAP
SEGMENT | pdf: <start>-<end> | printed: UNNUMBERED
SEGMENT | pdf: <start>-<end> | printed: <first_folio> .. <last_folio>
END

Worked example (a paper with an unnumbered title page, a 45-page body, and a 30-page appendix):
RECONCILED PAGE MAP
SEGMENT | pdf: 1-1 | printed: UNNUMBERED
SEGMENT | pdf: 2-46 | printed: 1 .. 45
SEGMENT | pdf: 47-76 | printed: A-1 .. A-30
END
"""

def _from_roman(s):
    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = prev = 0
    for ch in reversed(s.lower()):
        v = vals.get(ch, 0)
        total += -v if v < prev else v
        prev = max(prev, v)
    return total

def _parse_label(s):
    """'A-12' -> ('A-', 12, 'dec'); 'xi' -> ('', 11, 'roman-lower'); else None."""
    if not s:
        return None
    s = s.strip()
    m = re.fullmatch(r"([A-Za-z]{1,3}[-.\s]?)?0*(\d{1,4})", s)
    if m:
        return (m.group(1) or "", int(m.group(2)), "dec")
    if re.fullmatch(r"[ivxlcdm]{1,7}", s):
        return ("", _from_roman(s), "roman-lower")
    if re.fullmatch(r"[IVXLCDM]{1,7}", s):
        return ("", _from_roman(s), "roman-upper")
    return None

def _make_label(pref, num, style):
    if style == "roman-lower":
        return pref + _to_roman(num, upper=False)
    if style == "roman-upper":
        return pref + _to_roman(num, upper=True)
    return pref + str(num)

_SEG_RE = re.compile(r"SEGMENT\s*\|\s*pdf:\s*(\d+)\s*-\s*(\d+)\s*\|\s*printed:\s*(.+)", re.I)

def _parse_segments(raw):
    segs = []
    for line in raw.splitlines():
        m = _SEG_RE.search(line)
        if not m:
            continue
        a, b, printed = int(m.group(1)), int(m.group(2)), m.group(3).strip()
        if re.fullmatch(r"unnumbered", printed, re.I):
            segs.append({"a": a, "b": b, "unnumbered": True})
        else:
            parts = re.split(r"\s*\.\.\s*|\s+to\s+", printed)
            first = parts[0].strip()
            last = parts[-1].strip() if len(parts) > 1 else first
            segs.append({"a": a, "b": b, "unnumbered": False, "first": first, "last": last})
    return segs

def _labels_from_segments(segs, n):
    """Expand validated segments to per-physical-page labels, or (None, reason).
    Rejects gaps/overlaps/out-of-range, unparseable or style-mismatched endpoints,
    any segment whose folios do not increment by exactly 1, and incomplete coverage."""
    if not segs:
        return None, "no-segments"
    labels = [None] * n
    expected = 1
    for s in sorted(segs, key=lambda x: x["a"]):
        a, b = s["a"], s["b"]
        if a != expected or b < a or b > n:
            return None, "noncontiguous"
        expected = b + 1
        if s.get("unnumbered"):
            continue
        pf, pl_ = _parse_label(s.get("first")), _parse_label(s.get("last"))
        if pf is None or pl_ is None:
            return None, "unparseable-label"
        if pf[0] != pl_[0] or pf[2] != pl_[2]:
            return None, "style-mismatch"
        if pl_[1] - pf[1] != b - a:
            return None, "arithmetic-mismatch"
        for off, i in enumerate(range(a - 1, b)):
            labels[i] = _make_label(pf[0], pf[1] + off, pf[2])
    if expected != n + 1:
        return None, "incomplete-coverage"
    return labels, "ok"

def _annotate(m, note):
    m = dict(m)
    m["cross_check"] = (m.get("cross_check", "") + " | " + note).strip(" |")
    m["reconcile_note"] = note
    return m

def reconcile_with_flash(mech_map, pdf_path, call_fn):
    """Hand the mechanical map + the PDF to Flash, let it correct the map by reading
    the actual folios, then DETERMINISTICALLY validate (contiguous coverage,
    per-segment +1 monotonicity, >=85% agreement with the independent text-scan
    wherever the scan read a folio). Returns a reconciled map (same dict shape) on
    success, else the mechanical map untouched + a reconcile_note. call_fn(prompt,
    pdf_path) -> str is injected so this module stays decoupled from the Gemini client."""
    if not mech_map:
        return mech_map
    n = mech_map.get("n_pages") or 0
    if n <= 0:
        return _annotate(mech_map, "flash-reconcile skipped (no pages)")
    scan = mech_map.get("scan") or [None] * n
    hint = (render_details(mech_map) if mech_map.get("confidence") == "high"
            else "(no reliable mechanical map; read the printed folios off the pages yourself)")
    prompt = RECONCILE_PROMPT.replace("{{MECHANICAL}}", hint).replace("{{N}}", str(n))
    try:
        raw = call_fn(prompt, pdf_path) or ""
    except Exception as e:
        return _annotate(mech_map, f"flash-reconcile failed: {e}")
    labels, why = _labels_from_segments(_parse_segments(raw), n)
    if labels is None:
        return _annotate(mech_map, f"flash-reconcile rejected ({why})")
    # Ground-truth anchors = pages where the mechanical label and the INDEPENDENT
    # text-scan already agree (two sources concurring on the printed folio). Flash
    # must respect those (a uniform off-by-one disagrees with all of them -> reject).
    # Pages where the two disagree, or either is missing (title pages, mislabelled
    # front matter, header-only or image-only folios), are exactly what Flash is
    # here to adjudicate, so they are NOT held against it. This keeps raw-scan noise
    # (which misreads body numbers as folios) from rejecting a correct reconciliation.
    mech_labels = mech_map.get("labels") or [None] * n
    agree = tot = 0
    for i in range(n):
        s, ml = scan[i], mech_labels[i]
        if s is None or ml is None or _norm(s) != _norm(ml):
            continue
        tot += 1
        agree += 1 if _norm(labels[i]) == _norm(s) else 0
    if tot and agree / tot < 0.85:
        return _annotate(mech_map, f"flash-reconcile rejected (anchor-disagree {agree}/{tot})")
    return {"pdf": pdf_path, "n_pages": n, "method": "flash-reconciled",
            "confidence": "high",
            "cross_check": f"flash segments validated; scan-agree {agree}/{tot}",
            "labels": labels, "scan": scan,
            "sections": _sections(labels), "ceilings": _ceilings(labels),
            "mechanical_method": mech_map.get("method"),
            "flash_raw": raw.strip()[:2000]}

# ---------------------------------------------------------------- 05b citation guard
def _impossible(label, ceilings, max_dec):
    m = re.match(r"^([A-Za-z]{0,2}[-.]?)0*(\d{1,4})$", label)
    if not m:
        return False
    pref, num = m.group(1), int(m.group(2))
    if pref:
        return pref in ceilings and num > ceilings[pref]
    return max_dec is not None and num > max_dec

def validate_chk3_corrections(chk3_text, page_map):
    """Neutralise any 05b CORRECTION that proposes a page citation outside the
    document's actual page range (e.g. 'change A-13 to A-59' when the appendix
    ends at A-30). Deterministic; only fires on a HIGH-confidence map. Returns
    (cleaned_text, [rejections]). Off-by-one corrections WITHIN range are left
    alone (that needs the anchored remap, a separate step)."""
    if not chk3_text or not page_map or page_map.get("confidence") != "high":
        return chk3_text, []
    ceilings = page_map.get("ceilings", {})
    decimals = [int(x) for x in page_map.get("labels", []) if x and re.fullmatch(r"\d{1,4}", x)]
    max_dec = max(decimals) if decimals else None
    if not ceilings and max_dec is None:
        return chk3_text, []

    blocks = re.split(r"(?=^CITATION:)", chk3_text, flags=re.M)
    rejections = []
    out = []
    for seg in blocks:
        mstat = re.search(r"(?m)^STATUS:[ \t]*(.*)$", seg)
        mcorr = re.search(r"(?ms)^CORRECTION:[ \t]*(.*)\Z", seg)
        if mstat and mcorr and "VERIFIED" not in mstat.group(1).upper():
            corr = mcorr.group(1)
            cands = re.findall(r"(?<![\w])([A-Za-z]{1,2}[-.]?\d{1,4}|\d{1,4})(?![\w])", corr)
            bad = sorted({c for c in cands if _impossible(c, ceilings, max_dec)})
            if bad:
                rejections.append({"status": mstat.group(1).strip(), "bad": bad})
                seg = re.sub(r"(?m)^STATUS:.*$", "STATUS: VERIFIED", seg, count=1)
                seg = re.sub(
                    r"(?ms)^CORRECTION:.*\Z",
                    "CORRECTION: [auto-rejected: proposed page label(s) "
                    + ", ".join(bad) + " fall outside the document's actual page "
                    "range; original citation retained]", seg, count=1)
        out.append(seg)
    return "".join(out), rejections

# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python -m reviewer2.pagemap PAPER.pdf")
    m = build_page_map(sys.argv[1])
    print(f"{m['method']}/{m['confidence']} | {m['cross_check']}")
    print(render_details(m))
    print("ceilings:", m["ceilings"])
