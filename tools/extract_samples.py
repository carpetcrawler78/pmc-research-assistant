import pickle
import pandas as pd

with open('big_data/baseline/pickle/clean_articles.pkl', 'rb') as f:
    df = pickle.load(f)

mask = (
    df['keywords_filtered'].apply(len).between(3, 5) &
    df['abstract'].str.len().between(500, 1500) &
    df['title'].str.len().between(20, 120)
)

sample = df[mask].sample(10, random_state=None)[
    ['title', 'year', 'journal', 'abstract', 'keywords_filtered']
]

for i, row in sample.iterrows():
    print(f"--- Artikel {i} ---")
    print(f"TITLE   : {row['title']}")
    print(f"YEAR    : {row['year']}")
    print(f"JOURNAL : {row['journal']}")
    print(f"ABSTRACT: {row['abstract'][:300]}...")
    print(f"KEYWORDS: {row['keywords_filtered']}")
    print()