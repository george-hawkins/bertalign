# Rashomon parallel texts

The Japanese original and an English translation of Akutagawa Ryūnosuke's short
story *Rashōmon* (羅生門, 1915), used by [`../rashomon.py`](../rashomon.py)
to demonstrate Japanese↔English sentence alignment.

| File | Language | Paragraphs |
|------|----------|-----------|
| `rashomon_ja.txt` | Japanese (original) | 37 |
| `rashomon_en.txt` | English (Kojima, 1952) | 30 |

Each file is plain UTF-8 with one paragraph per line. The paragraph counts
differ because the translation groups the prose differently from the original;
that is harmless, since alignment is performed at the sentence level.

## Japanese — `rashomon_ja.txt`

* **Source:** [Aozora Bunko](https://www.aozora.gr.jp/), card 127
  ([work page](https://www.aozora.gr.jp/cards/000879/card127.html)), the "ruby"
  text edition
  ([127_ruby_150.zip](https://www.aozora.gr.jp/cards/000879/files/127_ruby_150.zip)),
  which is Shift-JIS encoded.
* **Rights:** public domain (Akutagawa died in 1927).
* **Cleaning** — [`clean_ja.py`](clean_ja.py) downloads the archive and:
  * decodes Shift-JIS to UTF-8;
  * removes Aozora markup — ruby readings `《…》`, the `｜` ruby-base marker, and
    `［＃…］` input notes — plus the leading legend block and the trailing
    colophon;
  * resolves the two "gaiji" (rare kanji that predate JIS X 0208 and so are
    recorded only as `［＃…面区点…］` descriptions) to their real Unicode
    characters, decoded from their JIS X 0213 men-ku-ten codes via the
    `euc_jis_2004` codec — `扭` (U+626D) and `眶` (U+7736).

## English — `rashomon_en.txt`

* **Source:** [Wikisource](https://en.wikisource.org/wiki/Rashomon_and_Other_Stories/Rashomon),
  translated by Kojima Takashi (1952), from *Rashomon and Other Stories*. This is
  the only Rashomon translation actually transcribed on Wikisource (a Glenn W.
  Shaw 1930 version is listed on the
  [versions page](https://en.wikisource.org/wiki/Rash%C5%8Dmon) but is not
  hosted there).
* **Rights:** hosted on Wikisource as public domain (US copyright not renewed).
* **Cleaning** — [`clean_en.py`](clean_en.py) downloads the rendered page and:
  * converts the HTML to text with `pandoc`;
  * drops the navigation/title header, the editorial footnote about the gate,
    the running-header lines (`[RASHOMON]` / `Rashomon`, from the scanned page
    images), and the zero-width-space page-break markers;
  * rejoins the two paragraphs that page breaks had split mid-sentence.

## Regenerating

```bash
uv run python clean_ja.py   # -> rashomon_ja.txt   (standard library only)
uv run python clean_en.py   # -> rashomon_en.txt   (requires pandoc on PATH)
```

Both scripts download their source afresh and write their output next to
themselves, so they fully reproduce the cleaned texts from the originals.
