import json
import time

import requests

# Your Semantic Scholar API key
API_KEY = "YOUR_SEMANTIC_SCHOLAR_API_KEY_HERE"

INPUT_FILE = "papers.json"
OUTPUT_BIB_FILE = "papers.bib"

HEADERS = {"x-api-key": API_KEY}


def query_semantic_scholar(title):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "limit": 1,
        "fields": "title,authors,venue,year,externalIds,url",
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        if data:
            return data[0]
    except Exception as e:
        print(f"Error querying {title}: {e}")
    return None


def create_bibtex(entry, paper_data):
    authors = " and ".join([a["name"] for a in paper_data.get("authors", [])])
    title = paper_data.get("title", "No Title")
    year = paper_data.get("year", entry.get("year"))
    venue = paper_data.get("venue", "")
    doi = paper_data.get("externalIds", {}).get("DOI", "")
    url = paper_data.get("url", entry.get("link", ""))
    note = entry.get("id", "")

    key = f"{authors.split()[0]}{year}"

    bibtex = "@article{"
    bibtex += f"{key},\n"
    bibtex += f"  title={{ {title} }},\n"
    if authors:
        bibtex += f"  author={{ {authors} }},\n"
    if venue:
        bibtex += f"  journal={{ {venue} }},\n"
    if year:
        bibtex += f"  year={{ {year} }},\n"
    if doi:
        bibtex += f"  doi={{ {doi} }},\n"
    if url:
        bibtex += f"  url={{ {url} }},\n"
    if note:
        bibtex += f"  note={{ {note} }},\n"
    bibtex += "}\n\n"
    return bibtex


# Load input JSON
with open(INPUT_FILE, encoding="utf-8") as f:
    papers = json.load(f)

all_bibtex = ""

for paper in papers:
    print(f"Processing: {paper['title']}")
    paper_data = query_semantic_scholar(paper["title"])
    if paper_data:
        bibtex_entry = create_bibtex(paper, paper_data)
        all_bibtex += bibtex_entry
    else:
        # fallback if no data found
        bibtex_entry = "@misc{"
        bibtex_entry += f"{paper['id']},\n  title={{ {paper['title']} }},\n  year={{ {paper['year']} }},\n  url={{ {paper['link']} }},\n  note={{ {paper['id']} }},\n"
        bibtex_entry += "}\n\n"
        all_bibtex += bibtex_entry
    time.sleep(1)  # avoid rate limit

# Save to .bib file
with open(OUTPUT_BIB_FILE, "w", encoding="utf-8") as f:
    f.write(all_bibtex)

print(f"\nBibTeX export complete: {OUTPUT_BIB_FILE}")
