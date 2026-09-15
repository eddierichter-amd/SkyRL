#!/usr/bin/env python3
"""Ensure installed vLLM can satisfy SkyRL's vLLM engine imports."""
import importlib.util
import sys

# Load the PyTorch ROCm runtime before vLLM and its optional compiler plugins.
import torch  # noqa: F401
import vllm

# At least one error-protocol path (vLLM moved openai.engine -> serve.engine).
ERROR_PROTOCOL_MODULES = [
    "vllm.entrypoints.openai.engine.protocol",
    "vllm.entrypoints.serve.engine.protocol",
]
REQUIRED = [
    "vllm.entrypoints.openai.chat_completion.protocol",
    "vllm.entrypoints.openai.completion.protocol",
    "vllm.entrypoints.openai.models.serving",
    "vllm.entrypoints.openai.chat_completion.serving",
    "vllm.entrypoints.openai.completion.serving",
]


def module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def main() -> int:
    print(f"vllm={vllm.__version__} path={vllm.__file__}")
    if not any(module_exists(m) for m in ERROR_PROTOCOL_MODULES):
        print("FAIL: missing ErrorResponse protocol module")
        return 1

    missing = [m for m in REQUIRED if not module_exists(m)]
    if missing:
        print("FAIL: missing modules:", ", ".join(missing))
        return 1

    try:
        from skyrl.backends.skyrl_train.inference_engines.vllm.vllm_import_compat import (
            ErrorInfo,
            ErrorResponse,
            openai_serving_chat_uses_online_renderer,
        )

        print(f"online_renderer_api={openai_serving_chat_uses_online_renderer()}")
        print(f"ErrorResponse={ErrorResponse} ErrorInfo={ErrorInfo}")
    except Exception as exc:  # noqa: BLE001 - report the complete compatibility failure
        print(f"FAIL: SkyRL vLLM compat imports: {exc}")
        return 1

    print("PASS: vLLM SkyRL API compatibility")
    return 0


if __name__ == "__main__":
    sys.exit(main())
