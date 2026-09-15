"""Import shims for vLLM OpenAI entrypoints across vLLM versions (incl. ROCm main builds)."""

__all__ = [
    "ErrorInfo",
    "ErrorResponse",
    "OpenAIServingRender",
    "openai_serving_chat_uses_online_renderer",
    "request_logger_class",
]

try:
    from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
except ImportError:
    from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse

try:
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender
except ImportError:
    try:
        from vllm.entrypoints.scale_out.render.serving import (
            ServingRender as OpenAIServingRender,
        )
    except ImportError:
        OpenAIServingRender = None  # type: ignore[misc, assignment]


def request_logger_class():
    try:
        from vllm.entrypoints.logger import RequestLogger
    except ImportError:
        from vllm.entrypoints.serve.utils.request_logger import RequestLogger
    return RequestLogger


def openai_serving_chat_uses_online_renderer() -> bool:
    import inspect

    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    return "online_renderer" in inspect.signature(OpenAIServingChat.__init__).parameters
