"""
Stub out heavy ML dependencies so tests run without GPU or downloaded models.
This file is loaded by pytest before any test module is imported.
"""

import sys
from unittest.mock import MagicMock

_STUBS = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.backends",
    "torch.backends.mps",
    "torchaudio",
    "transformers",
    "kokoro",
    "scipy",
    "scipy.signal",
    "tiktoken",
    "soundfile",
    "openai",
    "speech_recognition",
    "pydub",
    "pyaudio",
    "torchcodec",
]

for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
