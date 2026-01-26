#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        n = int(handler.headers.get("Content-Length", "0"))
    except Exception:
        n = 0
    raw = handler.rfile.read(max(0, n))
    try:
        return json.loads(raw.decode("utf-8")), None
    except Exception as e:
        return None, f"invalid_json: {e}"


class _State:
    def __init__(self, *, model_dir: Path, device: str, max_new_tokens: int, temperature: float) -> None:
        self.model_dir = model_dir
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.load_seconds = float(time.time() - t0)

        self.torch_cuda_available = bool(torch.cuda.is_available())

    def count_tokens(self, text: str) -> int:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return int(len(ids))

    def generate(self, prompt: str, *, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
        import torch  # type: ignore

        max_new_tokens = int(self.max_new_tokens if max_new_tokens is None else max_new_tokens)
        temperature = float(self.temperature if temperature is None else temperature)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        tokens_in = int(inputs["input_ids"].shape[-1])
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            torch.cuda.synchronize()

        t0 = time.time()
        out = self.model.generate(
            **inputs,
            do_sample=bool(temperature > 0),
            temperature=float(temperature),
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_total_ms = float((time.time() - t0) * 1000.0)

        out_ids = out[0].detach().cpu()
        tokens_out_total = int(out_ids.shape[-1])
        tokens_out = max(0, tokens_out_total - tokens_in)
        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)

        peak_gib = None
        if torch.cuda.is_available():
            peak_gib = float(torch.cuda.max_memory_allocated()) / (1024**3)

        return {
            "tokens_in": tokens_in,
            "tokens_out": int(tokens_out),
            "lat_total_ms": float(lat_total_ms),
            "cuda_peak_mem_gib": peak_gib,
            "output_text": text,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Local HF model dir (e.g., $BRACE_MODELS_ROOT/Mistral-7B-Instruct-v0.3)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18777)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise SystemExit(f"Missing --model-dir: {model_dir}")

    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "") else "auto"
    state = _State(model_dir=model_dir, device=device, max_new_tokens=int(args.max_new_tokens), temperature=float(args.temperature))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            payload, err = _read_json(self)
            if err is not None or payload is None:
                _json_response(self, 400, {"ok": False, "error": err})
                return

            if self.path == "/health":
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "model_dir": str(state.model_dir),
                        "load_seconds": state.load_seconds,
                        "torch_cuda_available": state.torch_cuda_available,
                    },
                )
                return

            if self.path == "/count_tokens":
                text = str(payload.get("text", ""))
                _json_response(self, 200, {"ok": True, "tokens": int(state.count_tokens(text))})
                return

            if self.path == "/generate":
                prompt = str(payload.get("prompt", ""))
                max_new_tokens = payload.get("max_new_tokens", None)
                temperature = payload.get("temperature", None)
                out = state.generate(
                    prompt,
                    max_new_tokens=int(max_new_tokens) if max_new_tokens is not None else None,
                    temperature=float(temperature) if temperature is not None else None,
                )
                out.update({"ok": True})
                _json_response(self, 200, out)
                return

            _json_response(self, 404, {"ok": False, "error": f"unknown_path: {self.path}"})

    server = ThreadingHTTPServer((str(args.host), int(args.port)), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
