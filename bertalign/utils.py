import re
from wtpsplit import SaT

# SaT is a multilingual segmenter, so a single model handles every language.
# It is comparatively expensive to load, so it is instantiated lazily and
# shared across all calls to split_sents.
_SAT_MODEL_NAME = "sat-3l-sm"
_sat_model = None


def _get_sat():
    global _sat_model
    if _sat_model is None:
        _sat_model = SaT(_SAT_MODEL_NAME)
    return _sat_model


def clean_text(text):
    cleaned = []
    text = text.strip()
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if line:
            line = re.sub(r"\s+", " ", line)
            cleaned.append(line)
    return "\n".join(cleaned)


def split_sents(text):
    # SaT is multilingual, so the segmenter works out the language itself; no
    # language code needs to be supplied.
    sents = _get_sat().split(text)
    return [sent.strip() for sent in sents if sent.strip()]


def yield_overlaps(lines, num_overlaps):
    lines = [_preprocess_line(line) for line in lines]
    for overlap in range(1, num_overlaps + 1):
        for out_line in _layer(lines, overlap):
            # check must be here so all outputs are unique
            out_line2 = out_line[
                :10000
            ]  # limit line so dont encode arbitrarily long sentences
            yield out_line2


def _layer(lines, num_overlaps, comb=" "):
    if num_overlaps < 1:
        raise Exception("num_overlaps must be >= 1")
    out = [
        "PAD",
    ] * min(num_overlaps - 1, len(lines))
    for ii in range(len(lines) - num_overlaps + 1):
        out.append(comb.join(lines[ii : ii + num_overlaps]))
    return out


def _preprocess_line(line):
    line = line.strip()
    if len(line) == 0:
        line = "BLANK_LINE"
    return line
