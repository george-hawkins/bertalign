"""Align Akutagawa's "Rashomon" (Japanese) with its English translation.

Reads the two cleaned source texts from texts/ and aligns them sentence by
sentence with Bertalign, printing each aligned pair. See texts/README.md for
where the texts come from and how they were prepared.
"""

from pathlib import Path

from bertalign import Bertalign

TEXTS = Path(__file__).parent / "texts"


def main():
    japanese = (TEXTS / "rashomon_ja.txt").read_text(encoding="utf-8")
    english = (TEXTS / "rashomon_en.txt").read_text(encoding="utf-8")

    aligner = Bertalign(japanese, english)
    aligner.align_sents()
    aligner.print_sents()


if __name__ == "__main__":
    main()
