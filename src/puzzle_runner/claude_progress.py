"""Readable terminal output for Claude's JSON event stream."""

from __future__ import annotations

import json


class ClaudeProgress:
    def __init__(self) -> None:
        self.streamed_blocks: set[int] = set()
        self.has_text = False
        self.line_open = False

    def __call__(self, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind == "stream_event":
            stream = event.get("event") or {}
            subtype = stream.get("type")
            if subtype == "message_start":
                self.streamed_blocks.clear()
            elif subtype == "content_block_delta":
                delta = stream.get("delta") or {}
                field = {"text_delta": "text", "thinking_delta": "thinking"}.get(delta.get("type"))
                if field and delta.get(field):
                    self.streamed_blocks.add(stream.get("index", 0))
                    self._write(delta[field])
                    self.has_text = self.has_text or field == "text"
            elif subtype == "content_block_stop":
                self._newline()
        elif kind in {"assistant", "user"}:
            content = (event.get("message") or {}).get("content", [])
            if not isinstance(content, list):
                return
            for index, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type in {"text", "thinking"} and kind == "assistant":
                    if index not in self.streamed_blocks:
                        self._write(block.get(block_type, ""))
                        self._newline()
                        self.has_text = self.has_text or block_type == "text"
                elif block_type == "tool_use":
                    self._newline()
                    self._write(f"Claude tool: {block.get('name', 'unknown')}\n")
                    inputs = block.get("input", {})
                    self._write(json.dumps(inputs, ensure_ascii=False, indent=2) + "\n")
                elif block_type == "tool_result":
                    self._newline()
                    label = "error" if block.get("is_error") else "result"
                    self._write(f"Claude tool {label}:\n")
                    result = block.get("content", "")
                    if isinstance(result, list):
                        result = "\n".join(
                            part.get("text", "") for part in result if isinstance(part, dict)
                        )
                    self._write(str(result))
                    self._newline()
            if kind == "assistant":
                self.streamed_blocks.clear()
        elif kind == "system":
            status = event.get("status") or event.get("subtype")
            if status:
                self._newline()
                self._write(f"Claude: {status}\n")
        elif kind == "result":
            self._newline()
            if not self.has_text and isinstance(event.get("result"), str):
                self._write(event["result"])
                self._newline()
            self._write(f"Claude: {event.get('subtype', 'finished')}\n")

    def _write(self, text: str) -> None:
        if text:
            print(text, end="", flush=True)
            self.line_open = not text.endswith("\n")

    def _newline(self) -> None:
        if self.line_open:
            self._write("\n")
