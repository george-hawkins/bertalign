"""Download the Kojima (1952) English "Rashomon" from Wikisource and write clean
prose.

Source: the only Rashomon translation actually hosted on Wikisource, by Kojima
Takashi (1952), from *Rashomon and Other Stories*:
    https://en.wikisource.org/wiki/Rashomon_and_Other_Stories/Rashomon

The page is scan-backed, so its rendered HTML carries transclusion artifacts.
We render the HTML to text with pandoc, then drop:
  * the navigation/title header and the editorial footnote about the gate;
  * running-header lines ([RASHOMON] / Rashomon, from the scanned page images);
  * zero-width-space (U+200B) page-break markers;
and rejoin the two paragraphs that page breaks split mid-sentence. A genuine
paragraph ends with terminal punctuation, so a fragment that begins with the
page-break marker AND follows a fragment lacking terminal punctuation is a
continuation.

Requires `pandoc` on PATH. Run it to regenerate rashomon_en.txt:
    python clean_en.py
"""

import re
import subprocess
import urllib.request
from pathlib import Path

URL = "https://en.wikisource.org/wiki/Rashomon_and_Other_Stories/Rashomon?action=render"
OUT = Path(__file__).parent / "rashomon_en.txt"

ZWSP = "​"  # zero-width space: Wikisource's page-break marker
TERMINAL = set(".!?…\"”’')）]")  # chars a complete paragraph may end with
RUNNING_HEADERS = {"[RASHOMON]", "Rashomon"}


def main():
    req = urllib.request.Request(
        URL, headers={"User-Agent": "bertalign-rashomon-demo/1.0"}
    )
    with urllib.request.urlopen(req) as resp:
        html = resp.read()
    text = subprocess.run(
        ["pandoc", "-f", "html", "-t", "plain", "--wrap=none"],
        input=html,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Body runs from just after the title (rendered "Rashomon^([1])") to just
    # before the editorial footnote (rendered "1.  ↑ ...").
    start = next(i for i, ln in enumerate(lines) if re.search(r"\^\(\[1\]\)", ln)) + 1
    end = next(i for i, ln in enumerate(lines) if re.match(r"\d+\.\s+↑", ln))

    paragraphs = []
    for ln in lines[start:end]:
        continuation = ln.startswith(ZWSP)
        cleaned = re.sub(r"\s{2,}", " ", ln.replace(ZWSP, "")).strip()
        if not cleaned or cleaned in RUNNING_HEADERS:
            continue
        if paragraphs and continuation and paragraphs[-1][-1] not in TERMINAL:
            paragraphs[-1] += " " + cleaned
        else:
            paragraphs.append(cleaned)

    OUT.write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name} ({len(paragraphs)} paragraphs)")


if __name__ == "__main__":
    main()
