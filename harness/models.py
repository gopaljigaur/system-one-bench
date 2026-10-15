"""The systems under test, and how to reach each one.

Every entry is either an HTTP server speaking `POST /v1/systemone` or a model
loaded in process. Both are reached through pydecide, which already speaks to
each of them, so nothing here reimplements a wire format.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class System:
    key: str
    label: str
    kind: str  # "server" or "local"
    params: str
    licence: str
    source: str
    build: Callable[[], object]
    note: str = ""


def _server(base_url: str) -> Callable[[], object]:
    def build() -> object:
        from decide import Client
        from decide.backends.typesafe import TypeSafeBackend

        return Client([TypeSafeBackend(api_key="unused", base_url=base_url, timeout=600)])

    return build


def _laya_mlx() -> object:
    from decide import Client
    from decide.backends.laya_mlx import LayaMLXBackend

    return Client([LayaMLXBackend()])


def _crossencoder() -> object:
    from decide import Client
    from decide.backends.crossencoder import CrossEncoderBackend

    return Client([CrossEncoderBackend("cross-encoder/ms-marco-MiniLM-L-6-v2")])


def _ollama(model: str) -> Callable[[], object]:
    def build() -> object:
        from decide import Client
        from decide.backends.llm import LLMBackend

        return Client(
            [
                LLMBackend(
                    base_url="http://localhost:11434/v1",
                    model=model,
                    api_key="ollama",
                    timeout=300,
                    max_tokens=256,
                )
            ]
        )

    return build


SYSTEMS: dict[str, System] = {
    "von": System(
        key="von",
        label="Von 1.3",
        kind="server",
        params="395M",
        licence="Apache-2.0",
        source="https://github.com/wfzyx/von",
        build=_server("http://127.0.0.1:8777"),
        note="von serve --device cpu --port 8777",
    ),
    "laya-serve": System(
        key="laya-serve",
        label="Laya 0.4.7 (laya-serve)",
        kind="server",
        params="421M",
        licence="Apache-2.0",
        source="https://github.com/NandhaKishorM/laya",
        build=_server("http://127.0.0.1:8779"),
        note="LAYA_PORT=8779 LAYA_DEVICE=cpu laya-serve",
    ),
    "laya-mlx": System(
        key="laya-mlx",
        label="Laya multilingual (MLX)",
        kind="local",
        params="322M",
        licence="Apache-2.0",
        source="https://huggingface.co/aac6fef/laya-multilingual-mlx",
        build=_laya_mlx,
        note="in process on Apple Silicon",
    ),
    "crossencoder": System(
        key="crossencoder",
        label="ms-marco MiniLM reranker",
        kind="local",
        params="22.7M",
        licence="Apache-2.0",
        source="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2",
        build=_crossencoder,
        note="a relevance scorer, included as a floor, not a decision model",
    ),
    "qwen3.8": System(
        key="qwen3.8",
        label="Qwen3.8 via ollama",
        kind="local",
        params="17 GB on disk",
        licence="see model card",
        source="https://ollama.com/library/qwen3.8",
        build=_ollama("qwen3.8:latest"),
        note="a chat model prompted for JSON, included as the LLM baseline",
    ),
    "llama3.2-1b": System(
        key="llama3.2-1b",
        label="Llama 3.2 1B via ollama",
        kind="local",
        params="1.2B",
        licence="Llama 3.2 licence",
        source="https://ollama.com/library/llama3.2",
        build=_ollama("llama3.2:1b"),
        note="small chat model, included to show where prompting stops working",
    ),
}
