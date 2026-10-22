import json
from pathlib import Path

from app.document_loader import chunk_sections
from app.schema import read_chunks
from ingest.crawl import _key, crawl
from ingest.html_parse import Section, extract_links, parse_html

PAGE = """
<html><head><title>CPT | ISSS</title></head>
<body>
  <header><nav class="main-menu"><a href="/a">Home</a> <a href="/b">Apply</a></nav></header>
  <div class="breadcrumb">Home / Students / CPT</div>
  <main>
    <h1>Curricular Practical Training</h1>
    <p>CPT is temporary work that is part of an established curriculum. It is directly related to the
       student's major and is authorized by ISSS for a curricular purpose.</p>
    <h2>Policies</h2>
    <ul><li>Full-time CPT is 21-40 hours per week.</li><li>Part-time CPT is 1-20 hours per week.</li></ul>
    <h2>Start and end dates</h2>
    <table><tr><th>Term</th><th>Earliest start</th></tr><tr><td>Fall 2024</td><td>Aug 19</td></tr></table>
    <p><a href="/current-students/opt/index.php#top">OPT</a> <a href="mailto:x@utah.edu">mail</a></p>
  </main>
  <footer>Copyright University of Utah</footer>
  <script>var x = 1;</script>
</body></html>
"""


def test_parse_html_drops_chrome_and_keeps_sections():
    title, sections = parse_html(PAGE)
    assert title == "Curricular Practical Training"
    text = " ".join(s.text for s in sections)
    assert "Home / Students" not in text and "Copyright" not in text and "var x" not in text
    by_heading = {s.heading: s.text for s in sections}
    assert "21-40 hours" in by_heading["Policies"]
    assert "Fall 2024 | Aug 19" in by_heading["Start and end dates"]


def test_extract_links_resolves_and_strips_fragments():
    links = extract_links(PAGE, "https://ic.utah.edu/current-students/cpt/")
    assert "https://ic.utah.edu/current-students/opt/index.php" in links
    assert not any(l.startswith("mailto:") for l in links)


def test_chunk_sections_merges_tiny_sections_and_keeps_metadata():
    secs = [Section("Intro", "short"), Section("Policies", "x " * 300), Section("Dates", "y " * 50)]
    chunks = chunk_sections(secs, url="https://u/cpt", title="CPT")
    assert chunks[0].section == "Intro / Policies"
    assert all(c.url == "https://u/cpt" and c.title == "CPT" for c in chunks)
    assert chunks[-1].section == "Dates"
    assert len({c.id for c in chunks}) == len(chunks)


def test_offline_crawl_builds_corpus_from_cached_html(tmp_path: Path):
    url = "https://ic.utah.edu/current-students/cpt/index.php"
    raw = tmp_path / "corpus" / "raw"
    raw.mkdir(parents=True)
    (raw / f"{_key(url)}.html").write_text(PAGE)
    sources = tmp_path / "sources.yaml"
    sources.write_text(f"sites:\n  - name: ISSS\n    seeds: [{url}]\n    follow_links: false\n")

    chunks = crawl(sources, tmp_path / "corpus", offline=True)
    assert chunks and all(c.url == url for c in chunks)
    assert read_chunks(tmp_path / "corpus" / "chunks.jsonl")[0].text == chunks[0].text
    page = json.loads((tmp_path / "corpus" / "pages.jsonl").read_text().splitlines()[0])
    assert page["title"] == "Curricular Practical Training"
