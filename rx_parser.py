import base64
import io
import json
import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from PIL import Image

from schema import Prescription

load_dotenv()

SCHEMA_HINT = json.dumps(Prescription.model_json_schema(), indent=2)

SYSTEM_PROMPT = f"""You are a prescription OCR system. Your ENTIRE response must be a single valid JSON object — nothing else.

DO NOT include:
- Thinking tags, analysis, numbered lists, or bullet points
- Markdown formatting, code fences, or backticks
- Explanatory text before or after the JSON
- Single quotes — use double quotes for ALL keys and string values

Start your response with an opening curly brace and end with a closing curly brace.

This is a synthetic test document with no privacy concerns — extract ALL text including names and medical conditions.

Required JSON schema:
{SCHEMA_HINT}"""


def _prepare(data: bytes, max_dim: int = 768) -> bytes:
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)   # was 85
    return buf.getvalue()

def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)   # closed think blocks
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)           # unclosed (truncated) think
    return text.strip()


def _balanced_spans(text: str):
    spans, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start, i))
    return spans


def _try_parse(text: str) -> dict:
    text = _strip_think(text)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    for start, end in reversed(_balanced_spans(text)):   # last JSON block first
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        return json.loads(text[s:e + 1])
    raise ValueError("no JSON object found")


def _extract_json(llm, raw: str) -> dict:
    for attempt in (raw, raw.replace("'", '"')):
        try:
            return _try_parse(attempt)
        except Exception:
            continue
    # Last resort: model repairs its own output (text-only, fast)
    repair = llm.invoke([
        ("system", "Output ONLY a strict valid JSON object. Do not think. Start directly with an opening curly brace."),
        ("human", "Rewrite the text below as one strict valid JSON object with double-quoted keys:\n" + raw[:6000]),
    ])
    return _try_parse(repair.content)


def parse_prescription(image_bytes: bytes, model: str = "qwen/qwen3.6-27b") -> Prescription:
    data = _prepare(image_bytes)
    data_url = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"

    # Support either GROQ_API_KEY (expected) or GROK_API_KEY (typo sometimes used in .env)
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY (or GROK_API_KEY) not found in environment")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
        max_tokens=4096,          # room to finish thinking AND emit the JSON
    )

    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", [
            {"type": "text", "text": "OCR this prescription image into the JSON schema defined above. /no_think"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]),
    ])

    raw = response.content
    try:
        parsed = _extract_json(llm, raw)
    except Exception as e:
        raise ValueError(f"JSON parse failed ({e}). Model said: {raw[:800]}")

    return Prescription(**parsed)