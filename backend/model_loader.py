from __future__ import annotations

import logging
from functools import lru_cache

from .model import CognitiveBiasModel

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model() -> CognitiveBiasModel:
    LOGGER.info("Loading cognitive bias model service")
    return CognitiveBiasModel()
