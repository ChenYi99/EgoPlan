"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from src.video_llama.video_llama.processors.base_processor import BaseProcessor
from src.video_llama.video_llama.processors.blip_processors import (
    Blip2ImageTrainProcessor,
    Blip2ImageEvalProcessor,
    BlipCaptionProcessor,
)
from src.video_llama.video_llama.processors.video_processor import (
    AlproVideoTrainProcessor,
    AlproVideoEvalProcessor
)

from src.video_llama.video_llama.processors.egoplan_video_processor import (
    EgoplanVideoTrainProcessor,
    EgoplanVideoEvalProcessor
)
from src.video_llama.video_llama.common.registry import registry

__all__ = [
    "BaseProcessor",
    "Blip2ImageTrainProcessor",
    "Blip2ImageEvalProcessor",
    "BlipCaptionProcessor",
    "AlproVideoTrainProcessor",
    "AlproVideoEvalProcessor",
    "EgoplanVideoTrainProcessor",
    "EgoplanVideoEvalProcessor"
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
