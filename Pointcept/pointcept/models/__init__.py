from .builder import build_model
from .default import DefaultSegmentor, DefaultClassifier

# Import only the backbones exercised by this reproducibility package.  The
# upstream package eagerly imports every research model (including optional
# instance-segmentation and pretraining stacks) from this module.  Besides
# adding unrelated dependencies, that makes a simple latency preflight spend
# minutes importing/initializing extensions that no experiment uses.
from .sparse_unet import *
from .point_transformer_v3 import *
from .octformer import *
