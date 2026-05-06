import requests
import json
from urllib.parse import quote

CROSSREF_API = "https://api.crossref.org/works"


def search_crossref(title):
    """Search CrossRef by title"""
    url = f"{CROSSREF_API}?query.title={quote(title)}&rows=1"

    response = requests.get(url)
    response.raise_for_status()

    items = response.json()["message"]["items"]

    if not items:
        return None

    return items[0]


def format_authors(authors):
    """Convert CrossRef author list to BibTeX format"""
    formatted = []
    for author in authors:
        given = author.get("given", "")
        family = author.get("family", "")
        formatted.append(f"{family}, {given}")
    return " and ".join(formatted)


def to_bibtex(metadata, id):
    """Convert CrossRef metadata to BibTeX"""
    title = metadata.get("title", [""])[0]
    authors = format_authors(metadata.get("author", []))
    journal = metadata.get("container-title", [""])[0]
    year = metadata.get("published-print", {}).get("date-parts", [[None]])[0][0]
    doi = metadata.get("DOI", "")

    bibtex = f"""
@article{{{doi.replace('/', '_')},
  title   = {{{title}}},
  author  = {{{authors}}},
  journal = {{{journal}}},
  year    = {{{year}}},
  doi     = {{{doi}}},
  note    = {{{id}}}
}}
"""
    return bibtex.strip()


# --------------------------
# Load your JSON
# --------------------------

with open("papers.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

bibliography = []

for paper in papers:
    print(f"Fetching: {paper['title']}")

    metadata = search_crossref(paper["title"])

    if metadata:
        bibtex_entry = to_bibtex(metadata, paper["id"])
        bibliography.append(bibtex_entry)
    else:
        print("❌ Not found in CrossRef")

# --------------------------
# Save BibTeX file
# --------------------------

with open("output.bib", "w", encoding="utf-8") as f:
    f.write("\n\n".join(bibliography))

print("\n✅ Bibliography saved to output.bib")
