from typing import Dict, List, Tuple


def _normalize_newline(raw_http: str) -> str:
    if not raw_http:
        return ""
    return raw_http.replace("#015#012", "\r\n").replace("\\r\\n", "\r\n")


def parse_http(raw_http: str) -> Dict:
    text = _normalize_newline(raw_http)
    if not text:
        return {
            "request_line": "",
            "method": "",
            "path_query": "",
            "protocol": "",
            "headers": {},
            "body": "",
        }

    sections = text.split("\r\n\r\n", 1)
    head = sections[0]
    body = sections[1] if len(sections) > 1 else ""
    lines: List[str] = [ln for ln in head.split("\r\n") if ln]
    if not lines:
        return {
            "request_line": "",
            "method": "",
            "path_query": "",
            "protocol": "",
            "headers": {},
            "body": body,
        }

    request_line = lines[0]
    method, path_query, protocol = _parse_request_line(request_line)
    headers = _parse_headers(lines[1:])
    return {
        "request_line": request_line,
        "method": method,
        "path_query": path_query,
        "protocol": protocol,
        "headers": headers,
        "body": body,
    }


def _parse_request_line(request_line: str) -> Tuple[str, str, str]:
    parts = request_line.split(" ")
    if len(parts) < 3:
        return "", "", ""
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def _parse_headers(header_lines: List[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for line in header_lines:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        headers[k.strip().lower()] = v.strip()
    return headers
