"""Patch vLLM's LoRA worker_manager to retry missing-adapter errors instead of
crashing the engine.

Context
-------
In async RL pipelines, the trainer writes broadcast directories (``step_N/``)
and deletes old ones via its cleanup policy. The inference worker receives
``load_lora_adapter`` requests that may reference a directory the trainer is
about to delete (or that NFS metadata hasn't propagated yet). Today vLLM raises
``LoRAAdapterNotFoundError`` once, which ``EngineCore`` treats as fatal and
kills the whole vllm process → 502 Bad Gateway → orchestrator aborts.

A short bounded retry absorbs both races cleanly. The "latest adapter" is
always still being written by the time a retry happens, so worst-case the
retry succeeds; if not, we still raise the original error.

Usage
-----
Invoke this as a Python -c script at container startup, before
``exec python3 -m dynamo.vllm …``. Safe to run multiple times — it checks for
the sentinel comment before patching.
"""

from __future__ import annotations

import glob
import sys


def _find_worker_manager() -> str:
    # Typical locations inside the DGD image; glob handles venv path drift.
    candidates = glob.glob(
        "/**/site-packages/vllm/lora/worker_manager.py", recursive=True
    )
    if not candidates:
        print("worker_manager.py not found — nothing to patch", file=sys.stderr)
        sys.exit(0)
    return candidates[0]


def _apply() -> None:
    path = _find_worker_manager()
    src = open(path).read()
    if "# prime-rl-retry" in src:
        print(f"already patched: {path}")
        return

    # Wrap _load_adapter. Strategy: rename original to _load_adapter_inner,
    # add a retrying dispatcher with the real name.
    needle_def = "    def _load_adapter(self, lora_request: LoRARequest) -> LoRAModel:"
    if needle_def not in src:
        print(f"unexpected signature in {path} — aborting", file=sys.stderr)
        sys.exit(1)

    replacement = (
        "    def _load_adapter(self, lora_request: LoRARequest) -> LoRAModel:  # prime-rl-retry\n"
        "        import time as _time\n"
        "        _last_exc = None\n"
        "        # Tight exponential-backoff on missing-adapter errors to absorb\n"
        "        # short-lived NFS / trainer-cleanup races without killing the engine.\n"
        "        for _attempt in range(6):  # total ~10s across retries\n"
        "            try:\n"
        "                return self._load_adapter_inner(lora_request)\n"
        "            except LoRAAdapterNotFoundError as _e:\n"
        "                _last_exc = _e\n"
        "                _time.sleep(0.5 * (2 ** _attempt))  # 0.5, 1, 2, 4, 8, 16\n"
        "        raise _last_exc\n"
        "\n"
        "    def _load_adapter_inner(self, lora_request: LoRARequest) -> LoRAModel:"
    )
    src = src.replace(needle_def, replacement, 1)
    open(path, "w").write(src)
    print(f"patched: {path}")


if __name__ == "__main__":
    _apply()
