"""
XML Structure Analysis
======================
1. Tag frequency statistics over N sampled articles from the archive
2. Tree view of a single XML file
3. Mapping: which tag/xpath → which paper column

Usage:
    python xml_structure_analysis.py               # full analysis, N=200
    python xml_structure_analysis.py --n 50        # sample 50 articles
    python xml_structure_analysis.py --tree        # print tree of first article only
"""

import tarfile, argparse, sys, re
from pathlib import Path
from collections import Counter, defaultdict

import lxml.etree as ET
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

CWD          = Path(__file__).parent
PROJECT_ROOT = CWD if (CWD / 'big_data').exists() else CWD.parent
BIG_DATA     = PROJECT_ROOT / 'big_data'

ARCHIVE = BIG_DATA / 'oa_noncomm_xml.PMC003xxxxxx.baseline.2026-01-23.tar.gz'
DTD_DIR = BIG_DATA / 'JATS-Archiving-1-3-MathML3-DTD'

BASE_URL = DTD_DIR.as_uri()
if not BASE_URL.endswith('/'):
    BASE_URL += '/'

# =============================================================================
# MAPPING: tag/xpath → paper column (from data_extraction.py)
# =============================================================================

TAG_COLUMN_MAP = {
    # Metadata
    '//article-id[@pub-id-type="pmc"]'  : 'pmcid',
    '//article-id[@pub-id-type="doi"]'  : 'doi',
    '//journal-title'                    : 'journal',
    '//pub-date//year'                   : 'year',
    '//contrib[@contrib-type="author"]'  : 'authors',
    '//article-title'                    : 'title',
    # Keywords
    '//kwd'                              : 'keywords',
    '//compound-kwd-part'                : 'keywords',
    '//nested-kwd'                       : 'keywords',
    # Abstract
    '//abstract[not(@abstract-type)]//p' : 'abstract',
    # Body sections
    '//body/sec'                         : 'introduction / methods / results / discussion / conclusion',
    # Tables
    '//table-wrap'                       : 'tables',
    # Noise
    'conflict/funding/acknowledgment'    : 'metadata_junk',
}

# Attributes that are used as filters/classifiers
ATTRIBUTE_FILTERS = {
    '@article-type'        : 'filter: only "research-article"',
    '@xml:lang'            : 'filter: only "en*"',
    '@pub-id-type'         : 'selects pmc vs doi id',
    '@abstract-type'       : 'prefer abstract without @abstract-type',
    '@contrib-type'        : 'filter: only "author"',
    'sec/title text'       : 'mapped via SECTION_MAP keywords',
}

# =============================================================================
# PARSER
# =============================================================================

_parser = ET.XMLParser(
    dtd_validation=False,   # faster for analysis
    load_dtd=True,
    recover=True,
    no_network=False
)

# =============================================================================
# SECTION 1: TAG FREQUENCY OVER N ARTICLES
# =============================================================================

def analyse_tag_frequency(archive_path: Path, n_sample: int = 200):
    """
    Sample N XML files from archive, count tag and attribute frequencies.
    Returns two DataFrames: tag_df, attr_df
    """
    tag_counter  = Counter()
    attr_counter = Counter()   # (tag, attr_name) pairs
    attr_val_counter = defaultdict(Counter)  # (tag, attr) → value counts

    n_parsed = 0

    print(f"\n{'='*60}")
    print(f" TAG FREQUENCY ANALYSIS  (sample: {n_sample} articles)")
    print(f"{'='*60}")

    with tarfile.open(archive_path, 'r:gz') as tar:
        for member in tar:
            if n_parsed >= n_sample:
                break
            if not (member.isfile() and member.name.lower().endswith(('.nxml', '.xml'))):
                continue
            f = tar.extractfile(member)
            if not f:
                continue
            try:
                root = ET.fromstring(f.read(), parser=_parser, base_url=BASE_URL)
            except Exception:
                continue

            for elem in root.iter():
                if not isinstance(elem.tag, str):
                    continue
                # Clean tag (remove namespace)
                tag = re.sub(r'\{[^}]+\}', '', elem.tag)
                tag_counter[tag] += 1
                for attr_name, attr_val in elem.attrib.items():
                    attr_name_clean = re.sub(r'\{[^}]+\}', '', attr_name)
                    key = f'{tag}[@{attr_name_clean}]'
                    attr_counter[key] += 1
                    attr_val_counter[key][attr_val] += 1

            n_parsed += 1
            if n_parsed % 50 == 0:
                print(f"  parsed {n_parsed}/{n_sample}...", file=sys.stderr)

    print(f"  Total articles parsed: {n_parsed}")

    # Build DataFrames
    tag_df = pd.DataFrame(
        tag_counter.most_common(),
        columns=['tag', 'count']
    )
    tag_df['per_article'] = (tag_df['count'] / n_parsed).round(2)
    tag_df['in_our_pipeline'] = tag_df['tag'].apply(
        lambda t: _tag_in_pipeline(t)
    )

    attr_df = pd.DataFrame(
        attr_counter.most_common(80),
        columns=['tag_attr', 'count']
    )
    attr_df['per_article'] = (attr_df['count'] / n_parsed).round(2)
    # Add top values
    attr_df['top_values'] = attr_df['tag_attr'].apply(
        lambda k: ', '.join(
            f'{v}({c})' for v, c in attr_val_counter[k].most_common(5)
        )
    )

    return tag_df, attr_df, attr_val_counter


def _tag_in_pipeline(tag: str) -> str:
    """Mark which pipeline column this tag feeds."""
    mapping = {
        'article-id'   : 'pmcid / doi',
        'journal-title': 'journal',
        'journal-id'   : 'journal',
        'pub-date'     : 'year',
        'year'         : 'year',
        'contrib'      : 'authors',
        'surname'      : 'authors',
        'given-names'  : 'authors',
        'article-title': 'title',
        'kwd'          : 'keywords',
        'compound-kwd-part': 'keywords',
        'nested-kwd'   : 'keywords',
        'abstract'     : 'abstract',
        'body'         : 'sections',
        'sec'          : 'sections',
        'table-wrap'   : 'tables',
        'th'           : 'tables',
        'td'           : 'tables',
        'xref'         : 'STRIPPED (citations)',
    }
    return mapping.get(tag, '')


# =============================================================================
# SECTION 2: TREE VIEW OF SINGLE XML
# =============================================================================

def print_xml_tree(root: ET.Element, max_depth: int = 6, max_children: int = 4,
                   indent: str = '', depth: int = 0):
    """
    Recursively print XML tree structure with tag, attributes, text preview.
    """
    if depth > max_depth:
        return

    if not isinstance(root.tag,str):
        return

    tag      = re.sub(r'\{[^}]+\}', '', root.tag)
    attrs    = ' '.join(f'{re.sub(r"{[^}]+}", "", k)}="{v}"'
                        for k, v in root.attrib.items())
    text_raw = (root.text or '').strip()[:60]
    text_str = f'  → "{text_raw}"' if text_raw else ''
    in_pipe  = _tag_in_pipeline(tag)
    pipe_str = f'  [→ {in_pipe}]' if in_pipe else ''

    print(f"{indent}<{tag}{' ' + attrs if attrs else ''}>{text_str}{pipe_str}")

    children = list(root)
    for i, child in enumerate(children):
        if i >= max_children:
            print(f"{indent}  ... ({len(children) - max_children} more children)")
            break
        print_xml_tree(child, max_depth, max_children,
                       indent + '  ', depth + 1)


def inspect_first_article(archive_path: Path):
    """Extract and print tree of the first valid XML in the archive."""
    print(f"\n{'='*60}")
    print(f" XML TREE VIEW (first article)")
    print(f"{'='*60}")

    with tarfile.open(archive_path, 'r:gz') as tar:
        for member in tar:
            if not (member.isfile() and member.name.lower().endswith(('.nxml', '.xml'))):
                continue
            f = tar.extractfile(member)
            if not f:
                continue
            try:
                root = ET.fromstring(f.read(), parser=_parser, base_url=BASE_URL)
                print(f"\nFile: {member.name}\n")
                print_xml_tree(root, max_depth=6, max_children=4)
                return
            except Exception as e:
                print(f"  parse error: {e}")
                continue


# =============================================================================
# SECTION 3: PIPELINE MAPPING SUMMARY
# =============================================================================

def print_pipeline_mapping():
    print(f"\n{'='*60}")
    print(f" PIPELINE MAPPING: XPath → paper column")
    print(f"{'='*60}\n")

    rows = []
    for xpath, col in TAG_COLUMN_MAP.items():
        rows.append({'xpath / tag': xpath, 'paper column': col})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print(f"\n{'─'*60}")
    print(f" ATTRIBUTE-BASED FILTERS & CLASSIFIERS\n")
    for attr, desc in ATTRIBUTE_FILTERS.items():
        print(f"  {attr:<30} {desc}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',    type=int, default=200, help='articles to sample')
    parser.add_argument('--tree', action='store_true',   help='tree view only')
    args = parser.parse_args()

    if not ARCHIVE.exists():
        sys.exit(f"Archive not found: {ARCHIVE}")

    # Always print mapping
    print_pipeline_mapping()

    if args.tree:
        inspect_first_article(ARCHIVE)
    else:
        # Tag frequency
        tag_df, attr_df, attr_val_counter = analyse_tag_frequency(ARCHIVE, args.n)

        print(f"\n{'='*60}")
        print(f" TOP 40 TAGS BY FREQUENCY")
        print(f"{'='*60}")
        print(tag_df.head(40).to_string(index=False))

        print(f"\n{'='*60}")
        print(f" TOP 40 TAG+ATTRIBUTE COMBINATIONS")
        print(f"{'='*60}")
        print(attr_df.head(40).to_string(index=False))

        # Save CSVs
        out_dir = PROJECT_ROOT / 'data'
        out_dir.mkdir(exist_ok=True)
        tag_df.to_csv(out_dir / 'xml_tag_frequencies.csv', index=False)
        attr_df.to_csv(out_dir / 'xml_attr_frequencies.csv', index=False)
        print(f"\nSaved: {out_dir / 'xml_tag_frequencies.csv'}")
        print(f"Saved: {out_dir / 'xml_attr_frequencies.csv'}")

        # Tree view of first article
        inspect_first_article(ARCHIVE)
