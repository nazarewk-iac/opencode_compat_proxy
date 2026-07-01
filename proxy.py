#!/usr/bin/env python3
"""OpenAI-compatible tool-call compatibility proxy.

Converts DeepSeek DSML and Qwen XML raw tool calls into standard
OpenAI tool_calls format. Sits between opencode (port 9526) and
llama.cpp (port 8000).
"""

import html
import json
import logging
import os
import re
import uuid

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("proxy")

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAM = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:9527")
HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("PROXY_PORT", "9526"))

HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "content-length", "content-type",
})

DSML_BAR = chr(0xFF5C)
DSML_OPEN = "<" + DSML_BAR + "DSML" + DSML_BAR + "tool_calls>"
DSML_CLOSE = "</" + DSML_BAR + "DSML" + DSML_BAR + "tool_calls>"
DSML_OPEN_ALT = "<|DSML|tool_calls>"
DSML_CLOSE_ALT = "</|DSML|tool_calls>"
DSML_TAG_PREFIX = "<" + DSML_BAR + "DSML" + DSML_BAR
DSML_CLOSE_PREFIX = "</" + DSML_BAR + "DSML" + DSML_BAR
DSML_TAG_PREFIX_ALT = "<|DSML|"
DSML_CLOSE_PREFIX_ALT = "</|DSML|"

RAW_TOOL_OPEN_MARKERS = (
    DSML_OPEN,
    "<DSML>tool_calls>",
    "<tool_calls>",
    "<tool_calls",
    "<tool_call>",
    "<tool_call",
)

RAW_TOOL_FRAGMENT_MARKERS = RAW_TOOL_OPEN_MARKERS + (
    DSML_CLOSE,
    DSML_TAG_PREFIX + "invoke",
    DSML_TAG_PREFIX + "parameter",
    DSML_TAG_PREFIX + "system-reminder",
    DSML_CLOSE_PREFIX + "invoke",
    DSML_CLOSE_PREFIX + "parameter",
    DSML_CLOSE_PREFIX + "tool_calls",
    DSML_CLOSE_PREFIX + "system-reminder",
    "<DSML:",
    "</DSML:",
    "<|DSML|invoke",
    "<|DSML|parameter",
    "<|DSML|system-reminder",
    "</|DSML|invoke",
    "</|DSML|parameter",
    "</|DSML|tool_calls",
    "</|DSML|system-reminder",
    "<dcp-system-reminder>",
    "</dcp-system-reminder>",
    "<dcp-message-id>",
    "</dcp-message-id>",
    "</tool_calls>",
    "</tool_call>",
)

INTERNAL_LEAK_SENTINELS = (
    "Active compressed blocks in this session:",
    "If your selected compression range includes any listed block",
    "required placeholder exactly once in the summary using",
)

_INTERNAL_ARTIFACT_BLOCK_PATTERNS = (
    re.compile(
        r"<dcp-system-reminder\b[^>]*>.*?(?:</dcp-system-reminder>|</\uff5cDSML\uff5csystem-reminder>|</\|DSML\|system-reminder>)",
        re.DOTALL,
    ),
    re.compile(
        r"<system-reminder\b[^>]*>.*?</system-reminder>",
        re.DOTALL,
    ),
    re.compile(
        r"<(?:\uff5cDSML\uff5c|\|DSML\|)system-reminder\b[^>]*>.*?</(?:\uff5cDSML\uff5c|\|DSML\|)system-reminder>",
        re.DOTALL,
    ),
    re.compile(
        r"<dcp-message-id\b[^>]*>.*?</dcp-message-id>",
        re.DOTALL,
    ),
)

SECTION_SIZE = 32
GUARD_SECTIONS = 2


def normalize_raw_tool_calls(text):
    """Normalize various DSML/Qwen XML formats to standard <｜DSML｜tool_calls> format."""
    bar = DSML_BAR
    text = _normalize_dsml_bars(text)
    # Format 1: <DSML>tool_calls> (DSML pseudo-namespace without bars)
    if "<DSML>tool_calls>" in text:
        text = text.replace("<DSML>tool_calls>", "<" + bar + "DSML" + bar + "tool_calls>", 1)
        text = re.sub(r'</DSML[:\s]+tool_calls\s*>', "</" + bar + "DSML" + bar + "tool_calls>", text)
        text = re.sub(r'<DSML[:\s]+(invoke)\s+', "<" + bar + "DSML" + bar + r"\1 ", text)
        text = re.sub(r'<DSML[:\s]+(parameter)\s+', "<" + bar + "DSML" + bar + r"\1 ", text)
        text = re.sub(r'</DSML[:\s]+(invoke|parameter)\s*>', "</" + bar + "DSML" + bar + r"\1>", text)
    # Format 2: <tool_calls> (bare XML, no DSML prefix)
    elif "<tool_calls>" in text and "</tool_calls>" in text:
        text = text.replace("<tool_calls>", "<" + bar + "DSML" + bar + "tool_calls>", 1)
        text = text.replace("</tool_calls>", "</" + bar + "DSML" + bar + "tool_calls>", 1)
        text = re.sub(r'<invoke\s+', "<" + bar + "DSML" + bar + "invoke ", text)
        text = re.sub(r'</invoke\s*>', "</" + bar + "DSML" + bar + "invoke>", text)
        text = re.sub(r'<parameter\s+', "<" + bar + "DSML" + bar + "parameter ", text)
        text = re.sub(r'</parameter\s*>', "</" + bar + "DSML" + bar + "parameter>", text)
    return text

app = FastAPI()


def strip_hop_by_hop_headers(headers):
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def normalize_arg_value(val):
    if isinstance(val, str):
        val = html.unescape(val)
        try:
            parsed = json.loads(val)
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass
        return val
    return json.dumps(val, ensure_ascii=False)


def make_tool_call(name, arguments, call_id=None):
    return {
        "id": call_id or "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {
            "name": name,
            "arguments": normalize_arg_value(arguments),
        },
    }


def _normalize_dsml_bars(text):
    return (
        text.replace(DSML_OPEN_ALT, DSML_OPEN)
        .replace(DSML_CLOSE_ALT, DSML_CLOSE)
        .replace(DSML_TAG_PREFIX_ALT + "invoke", DSML_TAG_PREFIX + "invoke")
        .replace(DSML_TAG_PREFIX_ALT + "parameter", DSML_TAG_PREFIX + "parameter")
        .replace(DSML_TAG_PREFIX_ALT + "system-reminder", DSML_TAG_PREFIX + "system-reminder")
        .replace(DSML_CLOSE_PREFIX_ALT + "invoke", DSML_CLOSE_PREFIX + "invoke")
        .replace(DSML_CLOSE_PREFIX_ALT + "parameter", DSML_CLOSE_PREFIX + "parameter")
        .replace(DSML_CLOSE_PREFIX_ALT + "system-reminder", DSML_CLOSE_PREFIX + "system-reminder")
    )


def has_complete_raw_tool_block(text):
    text = _normalize_dsml_bars(text)
    if DSML_OPEN in text and DSML_CLOSE in text:
        return True
    if "<DSML>tool_calls>" in text:
        return True
    if "<tool_calls>" in text and "</tool_calls>" in text:
        return True
    if "<tool_call>" in text and "</tool_call>" in text:
        return True
    return False


def _first_marker_index(text, markers):
    indexes = [idx for marker in markers if (idx := text.find(marker)) != -1]
    return min(indexes) if indexes else -1


def _orphan_fragment_start(text, marker_idx):
    for sentinel in INTERNAL_LEAK_SENTINELS:
        idx = text.rfind(sentinel, 0, marker_idx)
        if idx != -1:
            return text.rfind("\n", 0, idx) + 1
    return 0


def find_raw_tool_start(text):
    text = _normalize_dsml_bars(text)
    open_idx = _first_marker_index(text, RAW_TOOL_OPEN_MARKERS)
    fragment_idx = _first_marker_index(text, RAW_TOOL_FRAGMENT_MARKERS)
    if open_idx != -1 and (fragment_idx == -1 or open_idx <= fragment_idx):
        return open_idx
    if fragment_idx != -1:
        return _orphan_fragment_start(text, fragment_idx)
    return len(text)


def has_any_dsml_prefix(text):
    if not text:
        return False
    text = _normalize_dsml_bars(text)
    for marker in RAW_TOOL_FRAGMENT_MARKERS:
        if marker in text:
            return True
    tail = text[-150:] if len(text) > 150 else text
    for marker in RAW_TOOL_FRAGMENT_MARKERS:
        max_size = min(len(tail), len(marker) - 1)
        for size in range(max_size, 2, -1):
            if marker.startswith(tail[-size:]):
                return True
    return False


def parse_dsml_tool_calls(text):
    results = []
    for m in re.finditer(
        re.escape(DSML_OPEN) + r"(.*?)" + re.escape(DSML_CLOSE), text, re.DOTALL
    ):
        block = m.group(1)

        # Format 1: <name>fn</name><parameters>...</parameters>
        for tc in re.finditer(
            r"<name>\s*(.*?)\s*</name>.*?<parameters>\s*(.*?)\s*</parameters>",
            block,
            re.DOTALL,
        ):
            results.append(make_tool_call(tc.group(1).strip(), tc.group(2).strip()))

        # Format 2: <DSML invoke name="fn"><DSML parameter name="k" string="t">v</DSML parameter></DSML invoke>
        invoke_pat = re.escape("<" + DSML_BAR + "DSML" + DSML_BAR + "invoke") + r'\s+name="([^"]+)"\s*>'
        for inv in re.finditer(invoke_pat, block):
            fn_name = inv.group(1)
            after_invoke = block[inv.end():]
            end_invoke = re.search(re.escape("</" + DSML_BAR + "DSML" + DSML_BAR + "invoke>"), after_invoke)
            if not end_invoke:
                continue
            param_block = after_invoke[:end_invoke.start()]
            params = {}
            for p in re.finditer(
                re.escape("<" + DSML_BAR + "DSML" + DSML_BAR + "parameter") +
                r'\s+name="([^"]+)"\s*(?:string="[^"]*"\s*)?' +
                r">(.*?)</" + DSML_BAR + "DSML" + DSML_BAR + "parameter>",
                param_block,
                re.DOTALL,
            ):
                params[p.group(1)] = p.group(2).strip()
            results.append(make_tool_call(fn_name, json.dumps(params, ensure_ascii=False)))
    return results


def parse_qwen_xml_tool_calls(text):
    results = []
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        block = m.group(1)
        # Format A: <name>fn</name><parameters>...</parameters>
        name_m = re.search(r"<name>\s*(.*?)\s*</name>", block, re.DOTALL)
        args_m = re.search(r"<parameters>\s*(.*?)\s*</parameters>", block, re.DOTALL)
        if name_m:
            results.append(make_tool_call(
                name_m.group(1).strip(),
                (args_m.group(1).strip() if args_m else "{}"),
            ))
            continue
        # Format B: <function=fn><parameter=k>v</parameter></function>
        func_m = re.search(r"<function=([^>]+)>(.*?)</function>", block, re.DOTALL)
        if func_m:
            fn_name = func_m.group(1).strip()
            param_block = func_m.group(2)
            params = {}
            for p in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", param_block, re.DOTALL):
                params[p.group(1).strip()] = p.group(2).strip()
            results.append(make_tool_call(
                fn_name,
                json.dumps(params, ensure_ascii=False)
            ))
    return results


def parse_raw_tool_calls(text):
    tc = parse_dsml_tool_calls(text)
    if tc:
        return tc
    tc = parse_qwen_xml_tool_calls(text)
    if tc:
        return tc
    # tool block found but nothing parsed — log the block for debugging
    if has_complete_raw_tool_block(text):
        log.warning("parse_raw_tool_calls failed on: %s", text[:800])
    return []


def collect_text_fields(delta):
    parts = []
    for field in ("content", "reasoning_content", "reasoning"):
        val = delta.get(field)
        if isinstance(val, str) and val:
            parts.append(val)
    return "".join(parts)


def convert_non_streaming_response(body):
    msg = body.get("choices", [{}])[0].get("message", {})
    content = msg.get("content", "") or ""
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        log.info("Upstream already has tool_calls: %s", json.dumps(tool_calls, ensure_ascii=False)[:500])
    if not tool_calls and content:
        if DSML_CLOSE in content or "</tool_calls>" in content or "</tool_call>" in content:
            log.info("Raw tool close tag found in content (len=%d): %s", len(content), content[:500])
        if has_complete_raw_tool_block(content):
            normalized = normalize_raw_tool_calls(content)
            tool_calls = parse_raw_tool_calls(normalized)
            if tool_calls:
                log.info("Converted %d tool_calls from non-stream content", len(tool_calls))
                msg["tool_calls"] = tool_calls
                msg["content"] = None
                body["choices"][0]["finish_reason"] = "tool_calls"
            else:
                log.warning("has_complete_raw_tool_block true but parse empty! block=[%s]", content[:600])
    return body


def _strip_internal_artifacts_from_history_text(text):
    cleaned = text
    for pattern in _INTERNAL_ARTIFACT_BLOCK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


def _sanitize_content_text_parts(content):
    if isinstance(content, str):
        cleaned = _strip_internal_artifacts_from_history_text(content)
        return cleaned, cleaned != content
    if not isinstance(content, list):
        return content, False
    changed = False
    new_content = []
    for part in content:
        if not isinstance(part, dict):
            new_content.append(part)
            continue
        text_key = None
        for candidate in ("text", "input_text"):
            if isinstance(part.get(candidate), str):
                text_key = candidate
                break
        if text_key is None:
            new_content.append(part)
            continue
        cleaned = _strip_internal_artifacts_from_history_text(part[text_key])
        if cleaned != part[text_key]:
            changed = True
            if not cleaned.strip():
                continue
            new_part = dict(part)
            new_part[text_key] = cleaned
            new_content.append(new_part)
        else:
            new_content.append(part)
    return new_content, changed


def _sanitize_chat_internal_artifact_history(payload):
    if not isinstance(payload, dict):
        return
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    changed_parts = 0
    removed_messages = 0
    sanitized_messages = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized_messages.append(message)
            continue
        content = message.get("content")
        cleaned_content, changed = _sanitize_content_text_parts(content)
        if not changed:
            sanitized_messages.append(message)
            continue
        changed_parts += 1
        if isinstance(cleaned_content, str) and not cleaned_content.strip():
            removed_messages += 1
            continue
        if isinstance(cleaned_content, list) and not cleaned_content:
            removed_messages += 1
            continue
        new_message = dict(message)
        new_message["content"] = cleaned_content
        sanitized_messages.append(new_message)
    if changed_parts or removed_messages:
        payload["messages"] = sanitized_messages
        log.warning(
            "sanitized internal artifact history removed=%s changed=%s",
            removed_messages, changed_parts,
        )


def _normalize_assistant_messages(messages):
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            has_text_or_tool = False
            for part in content:
                if not isinstance(part, dict):
                    continue
                t = part.get("type")
                if t == "tool_use":
                    has_text_or_tool = True
                    break
                if t == "text" and str(part.get("text") or "").strip():
                    has_text_or_tool = True
                    break
            if not has_text_or_tool:
                content.append({"type": "text", "text": "."})
            continue
        if not content and not msg.get("tool_calls"):
            msg["content"] = "."


def _drop_empty_tools(payload):
    if not isinstance(payload, dict):
        return
    if isinstance(payload.get("tools"), list) and not payload["tools"]:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
    optional_params = payload.get("optional_params")
    if isinstance(optional_params, dict):
        if isinstance(optional_params.get("tools"), list) and not optional_params["tools"]:
            optional_params.pop("tools", None)


def _disable_responses_reasoning_merge(payload):
    if not isinstance(payload, dict):
        return
    payload["merge_reasoning_content_in_choices"] = False
    optional_params = payload.get("optional_params")
    if isinstance(optional_params, dict):
        optional_params["merge_reasoning_content_in_choices"] = False


def sse_json(line):
    prefix = "data: "
    if not line.startswith(prefix):
        return None
    payload = line[len(prefix):]
    if payload.strip() == "[DONE]":
        return {"done": True}
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return {"raw": payload, "done": False}


def build_stream_tool_call_chunks(tool_calls, chunk_id, model):
    chunks = []
    for tc in tool_calls:
        tc_id = tc["id"]
        fn = tc["function"]
        chunks.append({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": fn["name"], "arguments": ""},
                }]},
                "finish_reason": None,
            }],
        })
        args = fn["arguments"]
        chunk_size = 32
        for i in range(0, len(args), chunk_size):
            chunks.append({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "function": {"arguments": args[i:i + chunk_size]},
                    }]},
                    "finish_reason": None,
                }],
            })
    chunks.append({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls",
        }],
    })
    return chunks


def _make_content_sse(chunk_id, model, text):
    return "data: " + json.dumps({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text, "tool_calls": []}}],
    }, ensure_ascii=False) + "\n\n"


def _find_dsml_start(text):
    return find_raw_tool_start(text)


async def stream_with_sections(upstream_req, forwarded_for=""):
    chunk_id = "chatcmpl-" + uuid.uuid4().hex[:12]
    model = upstream_req.get("model", "deepseek")

    req_headers = {"Accept": "text/event-stream"}
    if forwarded_for:
        req_headers["x-forwarded-for"] = forwarded_for

    client = httpx.AsyncClient(timeout=None)
    try:
        resp = await client.send(
            client.build_request(
                "POST",
                UPSTREAM + "/v1/chat/completions",
                json=upstream_req,
                headers=req_headers,
            ),
            stream=True,
        )
    except Exception as e:
        await client.aclose()
        log.error("Failed to connect to upstream: %s", e)
        return JSONResponse(status_code=500, content={"error": {"message": f"Failed to connect to upstream: {e}", "type": "proxy_error"}})

    if resp.status_code != 200:
        err_content = await resp.aread()
        await resp.aclose()
        await client.aclose()
        log.error("Upstream returned error %d: %s", resp.status_code, err_content.decode("utf-8", errors="replace"))
        return Response(content=err_content, status_code=resp.status_code, media_type="application/json")

    async def generate():
        nonlocal chunk_id, model
        buffer = ""
        unflushed_text = ""
        pending = []
        dsml_mode = False
        content_collected = False
        thinking_state = 0  # 0: not started, 1: thinking, 2: finished

        try:
            async for raw_line in resp.aiter_lines():
                if not raw_line:
                    continue

                ev = sse_json(raw_line)
                if ev is None:
                    yield raw_line + "\n\n"
                    continue
                if ev.get("done"):
                    continue

                choices = ev.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})

                if ev.get("id"):
                    chunk_id = ev["id"]
                if ev.get("model"):
                    model = ev["model"]

                if "role" in delta:
                    yield raw_line + "\n\n"
                    continue

                reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                text = delta.get("content", "")

                is_reason = bool(reasoning)
                raw_chunk_text = reasoning if is_reason else text
                if not raw_chunk_text:
                    if not reasoning and not dsml_mode:
                        yield raw_line + "\n\n"
                    continue

                # Format chunk with think tags if transitioning
                chunk_text = ""
                if is_reason:
                    if thinking_state == 0:
                        chunk_text += "<think>\n"
                        thinking_state = 1
                    chunk_text += raw_chunk_text
                else:
                    if thinking_state == 1:
                        chunk_text += "\n</think>\n"
                        thinking_state = 2
                    chunk_text += raw_chunk_text

                buffer += chunk_text
                content_collected = True

                if dsml_mode:
                    if has_complete_raw_tool_block(buffer):
                        idx = _find_dsml_start(buffer)
                        if idx > 0:
                            prefix = buffer[:idx]
                            if thinking_state == 1:
                                prefix += "\n</think>\n"
                                thinking_state = 2
                            yield _make_content_sse(chunk_id, model, prefix)
                        tcs = parse_raw_tool_calls(normalize_raw_tool_calls(buffer))
                        if tcs:
                            for tc in build_stream_tool_call_chunks(tcs, chunk_id, model):
                                yield "data: " + json.dumps(tc) + "\n\n"
                        else:
                            log.warning("parse_raw_tool_calls returned empty for DSML block")
                        return
                    continue

                if has_any_dsml_prefix(buffer):
                    dsml_mode = True
                    for p in pending:
                        yield _make_content_sse(chunk_id, model, p)
                    pending.clear()
                    if unflushed_text:
                        yield _make_content_sse(chunk_id, model, unflushed_text)
                        unflushed_text = ""
                    continue

                unflushed_text += chunk_text
                while len(unflushed_text) >= SECTION_SIZE:
                    pending.append(unflushed_text[:SECTION_SIZE])
                    unflushed_text = unflushed_text[SECTION_SIZE:]
                    if len(pending) > GUARD_SECTIONS:
                        yield _make_content_sse(chunk_id, model, pending.pop(0))

            # Stream ended
            if thinking_state == 1:
                unflushed_text += "\n</think>\n"
                thinking_state = 2

            if dsml_mode:
                if content_collected:
                    idx = _find_dsml_start(buffer)
                    if idx > 0:
                        yield _make_content_sse(chunk_id, model, buffer[:idx])
                    if not has_complete_raw_tool_block(buffer) and idx < len(buffer):
                        yield _make_content_sse(chunk_id, model, buffer[idx:])
            else:
                for s in pending:
                    yield _make_content_sse(chunk_id, model, s)
                if unflushed_text:
                    yield _make_content_sse(chunk_id, model, unflushed_text)

            yield "data: " + json.dumps({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {"content": "", "tool_calls": []}, "finish_reason": "stop"}],
            }) + "\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.api_route("/v1/chat/completions", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(request: Request):
    body_bytes = await request.body()
    stream = False
    try:
        j = json.loads(body_bytes)
        if j:
            _sanitize_chat_internal_artifact_history(j)
            _normalize_assistant_messages(j.get("messages"))
            _drop_empty_tools(j)
            _disable_responses_reasoning_merge(j)
        if j and "tools" in j and isinstance(j["tools"], list):
            j["tools"] = [t for t in j["tools"] if isinstance(t, dict) and t.get("type") == "function"]
            if not j["tools"]:
                j.pop("tools")
        stream = j.get("stream", False)
        log.info("REQUEST: model=%s stream=%s msgs=%d tools=%d",
                 j.get("model", "?"), stream,
                 len(j.get("messages", [])),
                 len(j.get("tools", [])) if j and "tools" in j else 0)
        for i, msg in enumerate(j.get("messages", [])):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            tc = msg.get("tool_calls")
            log.info("  msg[%d] role=%s content_len=%d tool_calls=%s",
                     i, role, len(content or ""),
                     len(tc) if tc else 0)
    except Exception:
        j = None
        log.info("REQUEST (raw): %s", body_bytes.decode("utf-8", errors="replace")[:1000])

    client_host = request.client.host if request.client else "unknown"
    upstream_headers = strip_hop_by_hop_headers(dict(request.headers))
    upstream_headers.pop("host", None)
    upstream_headers["content-type"] = "application/json"
    if not upstream_headers.get("x-forwarded-for"):
        upstream_headers["x-forwarded-for"] = client_host

    if stream:
        if j is None:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    UPSTREAM + "/v1/chat/completions",
                    content=body_bytes,
                    headers=upstream_headers,
                    timeout=None,
                )
            return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))
        log.info("RESPONSE: streaming via stream_with_sections")
        return await stream_with_sections(j, client_host)

    async with httpx.AsyncClient() as client:
        upstream_resp = await client.post(
            UPSTREAM + "/v1/chat/completions",
            json=j if j is not None else json.loads(body_bytes),
            headers=upstream_headers,
            timeout=None,
        )

    try:
        result = upstream_resp.json()
        result = convert_non_streaming_response(result)
        msg = result.get("choices", [{}])[0].get("message", {})
        finish = result.get("choices", [{}])[0].get("finish_reason", "?")
        content_preview = (msg.get("content") or "")[:200].replace("\n", "\\n")
        if msg.get("tool_calls"):
            log.info("RESPONSE: non-stream tool_calls=%d finish=%s",
                     len(msg["tool_calls"]), finish)
        else:
            log.info("RESPONSE: non-stream finish=%s content=%s", finish, content_preview)
        return JSONResponse(content=result, status_code=upstream_resp.status_code)
    except Exception as e:
        log.info("RESPONSE: non-stream error=%s, raw_len=%d", e, len(upstream_resp.content))
        return Response(content=upstream_resp.content, status_code=upstream_resp.status_code, headers=dict(upstream_resp.headers))


@app.get("/v1/models")
async def models_list():
    async with httpx.AsyncClient() as client:
        resp = await client.get(UPSTREAM + "/v1/models", timeout=None)
    log.info("MODELS: %d models returned", len(resp.json().get("data", [])))
    return JSONResponse(content=resp.json(), status_code=resp.status_code)





@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all_proxy(request: Request, path: str):
    body_bytes = await request.body()
    log.info("CATCH-ALL %s /%s body=%s", request.method, path,
             body_bytes.decode("utf-8", errors="replace")[:500])
    upstream_headers = strip_hop_by_hop_headers(dict(request.headers))
    upstream_headers.pop("host", None)
    client_host = request.client.host if request.client else "unknown"
    if not upstream_headers.get("x-forwarded-for"):
        upstream_headers["x-forwarded-for"] = client_host
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            request.method,
            UPSTREAM + "/" + path,
            headers=upstream_headers,
            content=body_bytes,
            timeout=None,
        )
    log.info("CATCH-ALL %s /%s -> %d (len=%d)", request.method, path,
             resp.status_code, len(resp.content))
    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
