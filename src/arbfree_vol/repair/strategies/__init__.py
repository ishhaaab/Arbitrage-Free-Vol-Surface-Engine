"""Model-specific repair strategies for the repair pipeline.

Each strategy adapts one smile model (raw SVI, eSSVI, SABR) to the
``RepairStrategy`` protocol.  All model-package imports live in this
package, so ``repair.engine`` (the pipeline) has no knowledge of any
model API; adding a model family means adding one strategy module here.
"""
from arbfree_vol.repair.strategies._common import (
    RepairStrategy,
    _PathFitResult,
    _PrepStatus,
    _SlicePrep,
    _prepare_slice,
)
from arbfree_vol.repair.strategies.svi import SVIStrategy
from arbfree_vol.repair.strategies.essvi import ESSVIStrategy
from arbfree_vol.repair.strategies.sabr import SABRStrategy


def get_strategy(use_ssvi: bool = False, use_sabr: bool = False) -> RepairStrategy:
    """Select the repair strategy for the requested model path.

    ``use_ssvi`` and ``use_sabr`` are mutually exclusive.
    """
    if use_ssvi and use_sabr:
        raise ValueError("use_ssvi and use_sabr are mutually exclusive")
    if use_ssvi:
        return ESSVIStrategy()
    if use_sabr:
        return SABRStrategy()
    return SVIStrategy()
