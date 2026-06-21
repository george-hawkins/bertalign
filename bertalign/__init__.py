"""
Bertalign initialization
"""

__author__ = "Jason (bfsujason@163.com)"
__version__ = "1.1.0"

# wtpsplit pulls in skops, which at import time scans numpy/scipy types via
# pickle.whichmodule -- a scan that getattrs across every already-loaded module.
# If transformers is loaded first, that scan trips its lazy module loader into
# importing the Aria image processor, which hard-imports torchvision. Importing
# wtpsplit before sentence-transformers/transformers makes the scan run while
# transformers is absent, sidestepping the (otherwise spurious) torchvision dep.
import wtpsplit  # noqa: F401

from bertalign.encoder import Encoder

# See other cross-lingual embedding models at
# https://www.sbert.net/docs/pretrained_models.html

model_name = "LaBSE"
model = Encoder(model_name)

from bertalign.aligner import Bertalign
