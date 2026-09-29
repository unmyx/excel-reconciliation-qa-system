"""Shared pytest setup.

The project root is added to ``sys.path`` so tests import ``recon`` and
``scripts`` exactly the way the CLI does. Test fixtures for the reconciliation
logic are added as those modules land.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
