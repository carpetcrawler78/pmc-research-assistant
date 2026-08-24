"""
PMC XML Data Extraction Pipeline
==================================
Merged from: marlo-80_parser_2.py + 01_carpetcrawler78-data_extraction.ipynb

Output: raw_articles.pkl
Cleaning and EDA: see notebooks/02_cleaning_eda.ipynb
"""

import os, re, sys, tarfile
import pandas as pd
import lxml.etree as ET

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED, ALL_COMPLETED

# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_SUCCESS_LIMIT = 100_000   # stop after X successful articles
MAX_WORKERS       = 7         # use 7 of 8 cores
MAX_QUEUE_SIZE    = 100       # RAM protection

TESTRUN     = False   # True = max 10 articles (quick check)
SAVE_PICKLE = True   # True = save raw pickle (full run only!)
SAVE_CSV    = True   # True = save CSV

# Paths d
PROJECT_ROOT = Path(__file__).parent.parent
BIG_DATA     = PROJECT_ROOT / "big_data"
DATA_DIR     = PROJECT_ROOT / "data"

ARCHIVES = {
    "baseline": BIG_DATA / "oa_noncomm_xml.PMC003xxxxxx.baseline.2026-01-23.tar.gz",
    "incr":     BIG_DATA / "oa_noncomm_xml.incr.2026-03-04.tar.gz",
}
RUN_TYPE = "baseline"   # "baseline" or "incr"
FILEPATH = ARCHIVES[RUN_TYPE]

DTD_DIR = PROJECT_ROOT / "big_data" / "JATS-Archiving-1-3-MathML3-DTD"

# =============================================================================
# VALIDATION
# =============================================================================

if not DTD_DIR.exists():
    raise FileNotFoundError(str(f'DTD directory not found: {DTD_DIR}'))
if not FILEPATH.exists():
    raise FileNotFoundError(str(f"Archive not found: {FILEPATH}"))

BASE_URL = DTD_DIR.as_uri()
if not BASE_URL.endswith("/"):
    BASE_URL += "/"

print(f"Archive  : {FILEPATH.name}")
print(f"DTD URL  : {BASE_URL}")

# =============================================================================
# REGEX (precompiled for performance)
# =============================================================================

RE_BRACKETS_SQ = re.compile(r'\[[\s,;\d-]*\]')
RE_BRACKETS_RD = re.compile(r'\([\s,;\d-]*\)')
RE_WHITESPACE  = re.compile(r'\s{2,}')

# =============================================================================
# HELPER
# =============================================================================

def clean_text_formatting(text: str) -> str:
    """Remove citation brackets [1,2],  (3-5) , enclosing apostrophs when following/preceding the brackets and normalize whitespace."""
    if not text:
        return ""
    text = text.strip()
    if text.startswith("['") and text.endswith("']") or text.startswith("('") and text.endswith("')"):
        text = text[2:-2]
        #return re.sub(r'\s+', ' ', text).strip()
    elif text.startswith("[") and text.endswith("]") or text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return re.sub(r'\s+', ' ', text).strip()
    

# =============================================================================
# PARSER (module-level so worker processes can reuse it)
# =============================================================================

_parser = ET.XMLParser(dtd_validation=True, load_dtd=True, recover=True, no_network=False)

# =============================================================================
# WORKER
# =============================================================================

SECTION_MAP = {
    'introduction': ['introduction', 'background', 'context', 'rationale'],
    'methods':      ['method', 'material', 'implementation',
                     'experimental', 'procedure', 'methodology'],
    'results':      ['result', 'finding', 'observation', 'data analysis'],
    'discussion':   ['discussion', 'interpretation', 'comment'],
    'conclusion':   ['conclusion', 'summary', 'concluding', 'closing', 'outlook'],
}
NOISE_TITLES = {
    'conflict', 'funding', 'acknowledgment',
    'contribution', 'ethics', 'data availability'
}


def process_xml_worker(xml_content: bytes, filename: str):
    """
    Parse one XML file and return structured article data.

    Returns
    -------
    (paper_dict, None)   on success
    (None, error_dict)   on failure
    """
    try:
        root = ET.fromstring(xml_content, parser=_parser, base_url=BASE_URL)

        # Strip citations and MathML globally
        ET.strip_elements(root, 'xref', with_tail=True)
        ET.strip_elements(root, '{http://www.w3.org/1998/Math/MathML}*', with_tail=True)

        # Filter: research-article + English only
        article_type = root.get('article-type')
        lang_list    = root.xpath('//article/@xml:lang | //article-meta//@xml:lang')
        lang         = lang_list[0] if lang_list else "en"

        if article_type != "research-article" or not str(lang).startswith('en'):
            return None, {"filename": filename,
                          "error": f"not research-article: {article_type}"}

        paper = {col: "" for col in [
            'filename', 'pmcid', 'doi', 'journal', 'year', 'authors',
            'title', 'keywords', 'abstract',
            'introduction', 'methods', 'results', 'discussion', 'conclusion',
            'tables', 'metadata_junk'
        ]}
        paper['filename'] = filename.split('/')[-1]

        # Metadata — use "".join (not str()) for clean output
        paper['pmcid']   = clean_text_formatting(str(root.xpath('//article-id[@pub-id-type="pmc"]/text()')))
        paper['doi']     = clean_text_formatting(str(root.xpath('//article-id[@pub-id-type="doi"]/text()')))
        paper['journal'] = " ".join(root.xpath(
            '//journal-title/text() | //journal-id[@journal-id-type="nlm-ta"]/text()'))
        paper['year']    = clean_text_formatting(str(root.xpath('//pub-date//year/text()')[:1]))

        auth_list = [
            f'{str(a.xpath("./surname/text()"))}'
            f' {str(a.xpath("./given-names/text()"))}'.strip()
            for a in root.xpath('//contrib[@contrib-type="author"]//name')
        ]
        paper['authors'] = "; ".join(auth_list)

        # Title
        paper['title'] = clean_text_formatting(
            " ".join(root.xpath('//article-title//text()')))

        # Keywords — raw, no blacklist, no subj-group
        # (cleaning done in 02_cleaning_eda.ipynb)
        kwd_xpath = (
            '//kwd//text() | '
            '//compound-kwd-part//text() | '
            '//nested-kwd//text()'
        )
        raw_kwds = root.xpath(kwd_xpath)
        paper['keywords'] = "; ".join(sorted(set(
            k.strip() for k in raw_kwds if k.strip() and len(k.strip()) > 1
        )))

        if not paper['keywords']:
            return None, {"filename": filename, "error": "no keywords"}

        # Abstract — prefer main abstract (no @abstract-type), fallback to any
        abstract_paras = root.xpath('//abstract[not(@abstract-type)]//p')
        if not abstract_paras:
            abstract_paras = root.xpath('//abstract//p')
        paper['abstract'] = clean_text_formatting(
            ' '.join([' '.join(p.itertext()) for p in abstract_paras]))

        # Tables — plain text
        tables = []
        for table_wrap in root.xpath('//table-wrap'):
            try:
                headers   = [clean_text_formatting(th)
                             for th in table_wrap.xpath('.//th/text()')]
                cells     = [clean_text_formatting(td)
                             for td in table_wrap.xpath('.//td/text()')]
                table_txt = clean_text_formatting(
                    " ".join(headers) + " " + " ".join(cells))
                if table_txt:
                    tables.append(table_txt)
            except Exception:
                continue
        paper['tables'] = " ".join(tables)

        # Sections
        for sec in root.xpath('//body/sec'):
            sec_title = " ".join(sec.xpath('./title/text()')).lower()
            sec_text  = clean_text_formatting(
                " ".join([" ".join(p.itertext()) for p in sec.xpath('.//p')]))

            if any(noise in sec_title for noise in NOISE_TITLES):
                paper['metadata_junk'] += f"\n[{sec_title}]: {sec_text}"
                continue

            assigned = False
            for target, keywords in SECTION_MAP.items():
                if any(k in sec_title for k in keywords):
                    paper[target] = (paper[target] + " " + sec_text).strip()
                    assigned = True
            if not assigned:
                paper['metadata_junk'] += f"\n[unmapped: {sec_title}]: {sec_text}"

        # Quality check
        required = ['abstract', 'introduction', 'methods', 'results', 'discussion']
        for req in required:
            if not paper[req] or len(paper[req]) < 100:
                return None, {"filename": filename,
                              "error": f"section {req} missing/short"}

        return paper, None

    except Exception as e:
        return None, {"filename": filename, "error": str(e)}


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def run_extraction(archive_path: Path, max_articles: int = None):
    """
    Extract articles using multiprocessing.

    Parameters
    ----------
    archive_path : Path
    max_articles : int or None — None = full run, int = test limit

    Returns
    -------
    df, df_err : pd.DataFrame, pd.DataFrame
    """
    extracted_data, all_errors = [], []
    i = 0

    with ProcessPoolExecutor() as executor:
        # let the runtime decide how many workers to use (up to MAX_WORKERS) - it knows it better
        waiting_futures = list()

        with tarfile.open(archive_path, "r:gz") as tar:
            print(f"Starting extraction (limit: {max_articles or MAX_SUCCESS_LIMIT})...",
                  file=sys.stderr)

            for member in tar:
                i += 1
                if i % 500 == 0:
                    print(f"  {i} members | {len(extracted_data)} extracted",
                          file=sys.stderr)

                if max_articles and len(extracted_data) >= max_articles:
                    print(f"Test limit reached: {max_articles}", file=sys.stderr)
                    break
                if len(extracted_data) >= MAX_SUCCESS_LIMIT:
                    print(f"MAX_SUCCESS_LIMIT reached", file=sys.stderr)
                    break

                if member.isfile() and member.name.lower().endswith((".nxml", ".xml")):
                    f = tar.extractfile(member)
                    if f:
                        fut = executor.submit(process_xml_worker, f.read(), member.name)
                        fut.name = member.name
                        waiting_futures.append(fut)
                    else:
                        all_errors.append({"filename": member.name,
                                           "error": "could not extract from archive"})
                else:
                    all_errors.append({"filename": member.name, "error": "not xml/nxml"})

                # RAM protection: drain queue when full
                if len(waiting_futures) >= MAX_QUEUE_SIZE:
                    done, _ = wait(waiting_futures, return_when=FIRST_COMPLETED)
                    for fut in done:
                        res, err = fut.result()
                        if res: extracted_data.append(res)
                        if err: all_errors.append(err)
                        waiting_futures.remove(fut)

            # Drain remaining futures
            # for fut in as_completed(waiting_futures):
            #     res, err = fut.result()
            #     if res: extracted_data.append(res)
            #     if err: all_errors.append(err)
            done, _ = wait(waiting_futures, return_when=ALL_COMPLETED)
            for fut in done:
                res, err = fut.result()
                if res: extracted_data.append(res)
                if err: all_errors.append(err)
                waiting_futures.remove(fut)

            executor.shutdown(wait=True)


    df     = pd.DataFrame(extracted_data)
    df_err = pd.DataFrame(all_errors) if all_errors \
             else pd.DataFrame(columns=['filename', 'error'])

    total = len(df) + len(df_err)
    print(f"\nMembers processed  : {i}")
    print(f"Articles extracted : {len(df)}")
    print(f"Articles dropped   : {len(df_err)}")
    if total > 0:
        print(f"Success rate       : {100 * len(df) / total:.1f} %")

    return df, df_err


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    df, df_err = run_extraction(
        archive_path=FILEPATH,
        max_articles=10 if TESTRUN else None
    )

    if not df.empty:
        if SAVE_PICKLE:
            if TESTRUN:
                print("WARNING: TESTRUN=True — run full extraction before saving pickle.")
            else:
                out_pkl = BIG_DATA / "baseline" / "pickle" / "raw_articles.pkl"
                out_pkl.parent.mkdir(parents=True, exist_ok=True)
                df.to_pickle(out_pkl)
                print(f"Saved : {out_pkl}  ({len(df)} articles)")

        if SAVE_CSV and not TESTRUN:
            out_csv = BIG_DATA / "baseline" / "csv" / "raw_articles.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True) 
            df.to_csv(out_csv, index=False)
            print(f"Saved : {out_csv}")

    if not df_err.empty:
        err_path = DATA_DIR / "extraction_errors.csv"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        df_err.to_csv(err_path, index=False)
        print(f"Error log: {len(df_err)} entries → {err_path}")
