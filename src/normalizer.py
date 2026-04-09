import re
from typing import Dict, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit


UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
HEX_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")
TS_RE = re.compile(r"\b\d{10,13}\b")
NUM_RE = re.compile(r"\b\d+\b")


def normalize_value(text: str) -> str:
    if not text:
        return ""
    out = text
    out = UUID_RE.sub("<UUID>", out)
    out = HEX_RE.sub("<HEX>", out)
    out = TS_RE.sub("<TS>", out)
    out = NUM_RE.sub("<NUM>", out)
    return out


def normalize_path_query(path_query: str) -> Tuple[str, str]:
    if not path_query:
        return "", ""
    split = urlsplit(path_query)
    path = normalize_value(split.path)

    query_pairs = parse_qsl(split.query, keep_blank_values=True)
    query_pairs.sort(key=lambda x: x[0])
    norm_query_pairs = [(k, normalize_value(v)) for k, v in query_pairs]
    query = urlencode(norm_query_pairs, doseq=True)
    return path, query


def normalize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for k, v in headers.items():
        normalized[k.lower()] = normalize_value(v)
    return normalized
