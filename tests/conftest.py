import sys
from pathlib import Path

import langsmith as ls

# Tests must never export their local fixtures or prompts to an external
# tracing service. The explicit trace collector tests remain fully local.
ls.configure(enabled=False)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
