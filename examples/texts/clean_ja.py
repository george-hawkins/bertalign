"""Download 羅生門 (Rashomon) from Aozora Bunko and write clean UTF-8 prose.

Source: Aozora Bunko, 芥川龍之介 (Akutagawa Ryūnosuke), card 127, the "ruby" text
edition (Shift-JIS):
    https://www.aozora.gr.jp/cards/000879/files/127_ruby_150.zip

The edition uses Aozora's markup conventions, which this script removes:
  * 《…》  ruby (furigana) readings
  * ｜      marker for the start of a ruby base span
  * ［＃…］ input-side notes (e.g. gaiji descriptions, emphasis-dot positions)
plus the leading title/legend block and the trailing colophon.

Two rare kanji are "gaiji": characters that fall outside the old JIS X 0208 set,
so Aozora records them only as a ［＃…面区点…］ description rather than the
character itself. They do have Unicode codepoints, so we resolve each from its
JIS X 0213 men-ku-ten code via the euc_jis_2004 codec (no guessing).

Pure standard library. Run it to regenerate rashomon_ja.txt:
    python clean_ja.py
"""

import io
import re
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.aozora.gr.jp/cards/000879/files/127_ruby_150.zip"
OUT = Path(__file__).parent / "rashomon_ja.txt"


def gaiji(men, ku, ten):
    """The character for a JIS X 0213 men-ku-ten code, via EUC-JIS-2004."""
    if men == 1:
        raw = bytes([0xA0 + ku, 0xA0 + ten])
    else:  # plane 2 is prefixed with SS3 (0x8F) in EUC-JIS-2004
        raw = bytes([0x8F, 0xA0 + ku, 0xA0 + ten])
    return raw.decode("euc_jis_2004")


def main():
    req = urllib.request.Request(
        URL, headers={"User-Agent": "bertalign-rashomon-demo/1.0"}
    )
    with urllib.request.urlopen(req) as resp:
        archive = resp.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".txt"))
        raw = zf.read(name).decode("shift_jis")  # Aozora text is Shift-JIS

    lines = raw.splitlines()
    # The body lies between the second "----" divider (end of the legend block)
    # and the colophon, which starts with "底本：".
    dividers = [i for i, ln in enumerate(lines) if set(ln) == {"-"} and len(ln) > 10]
    start = dividers[1] + 1
    end = next(i for i, ln in enumerate(lines) if ln.startswith("底本："))
    body = "\n".join(lines[start:end]).strip()

    # Resolve the gaiji before stripping ［＃…］ notes; the reading that follows
    # each one in 《…》 is removed by the generic ruby strip below.
    body = re.sub(r"※［＃「てへん＋丑」[^］]*］", gaiji(2, 12, 93), body)  # ねじ
    body = re.sub(r"※［＃「目＋匡」[^］]*］", gaiji(1, 88, 81), body)  # まぶた

    body = re.sub(r"［＃[^］]*］", "", body)  # input notes
    body = body.replace("｜", "")  # ruby-base start marker
    body = re.sub(r"《[^》]*》", "", body)  # ruby readings

    paragraphs = [ln.strip() for ln in body.splitlines() if ln.strip()]
    OUT.write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name} ({len(paragraphs)} paragraphs)")


if __name__ == "__main__":
    main()
