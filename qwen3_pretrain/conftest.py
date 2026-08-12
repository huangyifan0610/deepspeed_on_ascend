"""Make qwen3_pretrain importable for pytest regardless of the cwd."""

# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
