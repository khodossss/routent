"""Suppress noisy third-party warnings.

Import this module BEFORE transformers/sentence_transformers to silence:
- transformers model load reports (BertModel LOAD REPORT)
- max_length/max_new_tokens warnings
- generation_config deprecation
- sentence-transformers FutureWarning
- HuggingFace hub symlink warning on Windows
"""

import os
import warnings

# Disable HF symlink warning before import
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Disable load reports (BertModel LOAD REPORT table)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Disable progress bars ("Loading weights: 100%|...")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# Prevent Windows tokenizer subprocess issues
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Offline mode: skip all HF Hub network calls if models are already cached.
# Enable by setting ROUTENT_HF_OFFLINE=1 in .env when HF Hub is down or
# when you want to avoid update-check pings for cached models.
if os.environ.get("ROUTENT_HF_OFFLINE", "0") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Silence Python warnings at library level
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")

# Silence transformers logger once available
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
except ImportError:
    pass
