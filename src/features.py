import json
import math
import re
from typing import Dict, List, Tuple
from urllib.parse import parse_qsl

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler

from src.http_parser import parse_http
from src.normalizer import normalize_headers, normalize_value


# ==================== 1. URL模板解析器 ====================
class UrlTemplateParser:
    """
    URL模板解析器 - 将动态值替换为语义占位符
    用于将原始 path+query 中的 ID、UUID、日期等具体值泛化为模板
    """
    PATTERNS = [
        (r'\b\d{8,}\b', '<NUM>'),                     # 长数字（ID）
        (r'\b\d{4}-\d{2}-\d{2}\b', '<DATE>'),         # 日期格式
        (r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '<UUID>', re.I),
        (r'\b[a-f0-9]{32}\b', '<MD5>', re.I),
        (r'\b[a-zA-Z0-9_\-]{20,}\b', '<TOKEN>'),
        (r'\b\d+(?:\.\d+)+\b', '<VERSION>'),          # 版本号 1.0, 2.1.3
        (r'\b\d+\b', '<NUM>'),
    ]

    @classmethod
    def parse(cls, url: str) -> str:
        """将URL中的动态值替换为占位符，保留原始结构"""
        if not url or not url.strip():
            return ""
        result = url
        for pattern, repl, *flags in cls.PATTERNS:
            flag = flags[0] if flags else 0
            if flag:
                result = re.sub(pattern, repl, result, flags=flag)
            else:
                result = re.sub(pattern, repl, result)
        return result


# ==================== 2. URL解析函数 ====================
def parse_url_from_log(rec: Dict) -> Tuple[str, str, str]:
    """
    从日志中提取 method, path, query
    优先级：http请求行 > uri + http补充 > 仅uri
    """
    http_raw = rec.get("http", "")
    method = rec.get("method", "GET").upper()
    path = ""
    query = ""
    
    # 1. 尝试从 http 字段解析完整URL
    if http_raw:
        # 解码转义字符
        http_decoded = http_raw.replace("\\/", "/").replace("\\r\\n", "\r\n")
        
        # 提取请求行（第一行）
        request_line = http_decoded.split("\r\n")[0] if "\r\n" in http_decoded else http_decoded
        
        # 检查是否为有效HTTP请求行
        if " " in request_line and ("HTTP/" in request_line or request_line.startswith(("GET ", "POST ", "PUT ", "DELETE ", "HEAD ", "OPTIONS "))):
            parts = request_line.split(" ")
            if len(parts) >= 2:
                method = parts[0].upper()
                full_url = parts[1]
                
                # 分离路径和查询参数
                if "?" in full_url:
                    path, query = full_url.split("?", 1)
                else:
                    path, query = full_url, ""
    
    # 2. 如果 path 为空，回退到 uri 字段
    if not path:
        uri_raw = rec.get("uri", "")
        if uri_raw:
            path = uri_raw.replace("\\/", "/")
            # 尝试从 http 中提取查询参数
            if http_raw and "?" in http_raw:
                import re
                match = re.search(r'\?([^\\\s"\']+)', http_raw)
                if match:
                    query = match.group(1)
    
    # 3. 解码路径中的反斜杠（最终确保路径是标准格式）
    path = path.replace("\\/", "/")
    
    return method, path, query


# ==================== 3. URL模式特征提取 ====================
def extract_url_pattern_features(path: str, query: str) -> List[float]:
    """提取URL结构模式特征 - 17维"""
    segments = [s for s in path.split('/') if s]
    depth = len(segments)

    seg_types = [_classify_segment(s) for s in segments]
    type_counts = {
        'word': seg_types.count('word'),
        'digit': seg_types.count('digit'),
        'version': seg_types.count('version'),
        'uuid': seg_types.count('uuid'),
        'mixed': seg_types.count('mixed'),
        'param': seg_types.count('param'),
    }

    features = [
        float(depth),
        float(type_counts['digit']),
        float(1 if type_counts['version'] > 0 else 0),
        float(1 if type_counts['uuid'] > 0 else 0),
    ]

    if depth >= 1:
        features.append(1.0 if _classify_segment(segments[-1]) in ['digit', 'uuid', 'param'] else 0.0)
    else:
        features.append(0.0)
    if depth >= 2:
        features.append(1.0 if _classify_segment(segments[-2]) == 'word' else 0.0)
    else:
        features.append(0.0)

    # 查询参数特征
    query_params = parse_qsl(query, keep_blank_values=True) if query else []
    query_keys = [k for k, _ in query_params]
    query_values = [v for _, v in query_params]

    features.extend([
        float(len(query_params)),
        float(len(set(query_keys))),
        float(any(v.isdigit() for v in query_values)),
        float(any(v.replace('.', '').isdigit() and '.' in v for v in query_values)),
    ])

    # API版本号
    version = _extract_api_version(segments)
    features.extend([
        1.0 if version else 0.0,
        float(int(version.split('.')[0])) if version else 0.0,
    ])

    features.extend([
        float(type_counts['word']),
        float(type_counts['mixed']),
        float(type_counts['param']),
    ])

    total_seg = depth if depth > 0 else 1
    type_freq = [c / total_seg for c in type_counts.values() if c > 0]
    entropy = -sum(f * math.log(f) for f in type_freq) if type_freq else 0.0
    features.append(entropy)

    max_seg_len = max((len(s) for s in segments), default=0)
    features.append(float(max_seg_len))

    return features


def _classify_segment(segment: str) -> str:
    if not segment:
        return 'empty'
    if segment.startswith(':') or segment.startswith('{') or segment.startswith('<'):
        return 'param'
    if segment.isdigit():
        return 'digit'
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', segment, re.I):
        return 'uuid'
    if re.match(r'^v?\d+(?:\.\d+)*$', segment):
        return 'version'
    if re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', segment):
        return 'word'
    return 'mixed'


def _extract_api_version(segments: List[str]) -> str:
    for seg in segments:
        if re.match(r'^v?\d+(?:\.\d+)*$', seg):
            return seg[1:] if seg[0].lower() == 'v' else seg
    return ""


# ==================== 4. 增强特征提取 ====================
def extract_enhanced_features(
    path: str,
    query: str,
    headers: Dict[str, str],
    body: str,
    method: str,
    param_pairs: List[Tuple[str, str]]
) -> List[float]:
    """增强版无监督特征提取 - 52维"""
    merged = f"{path} {query} {body}".strip()

    # 基础统计特征 (14维)
    basic_features = [
        float(len(path)),
        float(len(query)),
        float(len(headers)),
        float(len(body)),
        float(len(headers.get("cookie", ""))),
        _ratio_digit(merged),
        _ratio_special(merged),
        _ratio_upper(merged),
        float(merged.count("=")),
        float(merged.count("&")),
        float(merged.count("/")),
        float(merged.count(";")),
        float(merged.count("%")),
        float(merged.count("\n") + merged.count("\\n")),
    ]

    # 参数特征 (5维)
    key_lengths = [len(k) for k, _ in param_pairs]
    val_lengths = [len(v) for _, v in param_pairs]

    param_features = [
        float(len(param_pairs)),
        float(len(set(k for k, _ in param_pairs))),
        float(np.mean(key_lengths)) if key_lengths else 0.0,
        float(np.std(key_lengths)) if len(key_lengths) > 1 else 0.0,
        float(np.mean(val_lengths)) if val_lengths else 0.0,
    ]

    # JSON结构特征 (13维)
    json_features = _extract_json_features(body)

    # 客户端特征 (3维)
    user_agent = headers.get("user-agent", "").lower()
    merged_lower = merged.lower()

    client_features = [
        1.0 if "headless" in user_agent else 0.0,
        1.0 if any(bot in user_agent for bot in ["python-requests", "curl", "wget", "go-http-client"]) else 0.0,
        1.0 if any(plugin in merged_lower for plugin in ["gadget", "eazybi", "plugin", "rest"]) else 0.0,
    ]

    # URL模式特征 (17维)
    url_pattern_features = extract_url_pattern_features(path, query)

    return basic_features + param_features + json_features + client_features + url_pattern_features


def _extract_json_features(body: str) -> List[float]:
    """提取JSON结构特征 - 13维"""
    if not body or not body.strip():
        return [0.0] * 13

    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return [0.0] * 13

    string_lengths = []
    has_line_break = 0
    has_sql_comment = 0
    special_chars = set('!@#$%^&*()_+={}[]|\\:;"\'<>,.?/~`')
    special_char_total = 0
    digit_total = 0
    uppercase_total = 0
    total_chars = 0
    keys = []
    array_lengths = []
    array_count = 0

    def traverse(value):
        nonlocal has_line_break, has_sql_comment, special_char_total
        nonlocal digit_total, uppercase_total, total_chars, array_count

        if isinstance(value, str):
            string_lengths.append(len(value))
            if '\n' in value or '\\n' in value:
                has_line_break = 1
            if '--' in value:
                has_sql_comment = 1
            for ch in value:
                total_chars += 1
                if ch.isdigit():
                    digit_total += 1
                elif ch.isupper():
                    uppercase_total += 1
                elif ch in special_chars:
                    special_char_total += 1
        elif isinstance(value, dict):
            for k, v in value.items():
                keys.append(k)
                traverse(v)
        elif isinstance(value, list):
            array_count += 1
            array_lengths.append(len(value))
            for item in value:
                traverse(item)

    traverse(obj)

    max_string_len = float(max(string_lengths)) if string_lengths else 0.0
    avg_string_len = float(sum(string_lengths) / len(string_lengths)) if string_lengths else 0.0
    string_len_std = float(np.std(string_lengths)) if len(string_lengths) > 1 else 0.0

    numeric_ratio = digit_total / total_chars if total_chars > 0 else 0.0
    uppercase_ratio = uppercase_total / total_chars if total_chars > 0 else 0.0

    key_count = float(len(keys))
    key_lengths = [len(k) for k in keys]
    max_key_len = float(max(key_lengths)) if key_lengths else 0.0
    avg_key_len = float(sum(key_lengths) / len(key_lengths)) if key_lengths else 0.0

    max_array_len = float(max(array_lengths)) if array_lengths else 0.0

    return [
        max_string_len, avg_string_len, string_len_std,
        float(has_line_break), float(has_sql_comment),
        float(special_char_total), numeric_ratio, uppercase_ratio,
        key_count, max_key_len, avg_key_len, max_array_len, float(array_count),
    ]


def _ratio_digit(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isdigit() for ch in text) / len(text)


def _ratio_upper(text: str) -> float:
    if not text:
        return 0.0
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return 0.0
    return sum(ch.isupper() for ch in alpha) / len(alpha)


def _ratio_special(text: str) -> float:
    if not text:
        return 0.0
    special = re.findall(r"[^a-zA-Z0-9\s]", text)
    return len(special) / len(text)


# ==================== 5. 参数对提取 ====================
def extract_param_pairs(method: str, query: str, body: str, headers: Dict[str, str]) -> List[Tuple[str, str]]:
    """提取参数键值对"""
    if method == "GET":
        pairs = _pairs_from_query(query)
        return pairs if pairs else _pairs_from_body(body, headers)
    pairs = _pairs_from_body(body, headers)
    return pairs if pairs else _pairs_from_query(query)


def _pairs_from_query(query: str) -> List[Tuple[str, str]]:
    if not query:
        return []
    pairs = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True)]
    pairs.sort(key=lambda x: x[0])
    return pairs


def _pairs_from_body(body: str, headers: Dict[str, str]) -> List[Tuple[str, str]]:
    text = (body or "").strip()
    if not text:
        return []
    content_type = headers.get("content-type", "").lower()

    if "application/json" in content_type or text.startswith("{") or text.startswith("["):
        pairs = _pairs_from_json_body(text)
        if pairs:
            return pairs
    if "application/x-www-form-urlencoded" in content_type or "=" in text:
        pairs = _pairs_from_form_body(text)
        if pairs:
            return pairs
    return []


def _pairs_from_form_body(body: str) -> List[Tuple[str, str]]:
    try:
        pairs = [(k, v) for k, v in parse_qsl(body, keep_blank_values=True)]
    except Exception:
        return []
    pairs.sort(key=lambda x: x[0])
    return pairs


def _pairs_from_json_body(body: str) -> List[Tuple[str, str]]:
    try:
        obj = json.loads(body)
    except Exception:
        return []
    out: List[Tuple[str, str]] = []
    _flatten_json(obj, "", out)
    out.sort(key=lambda x: x[0])
    return out


def _flatten_json(value, prefix: str, out: List[Tuple[str, str]]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            next_prefix = f"{prefix}.{k}" if prefix else str(k)
            _flatten_json(v, next_prefix, out)
    elif isinstance(value, list):
        for idx, v in enumerate(value):
            next_prefix = f"{prefix}[{idx}]"
            _flatten_json(v, next_prefix, out)
    else:
        out.append((prefix or "_root", "" if value is None else str(value)))


# ==================== 6. 主特征构建器 ====================
class FeatureBuilder:
    """单条日志特征提取器 - 无监督学习版本"""

    def __init__(self, feature_cfg: Dict):
        analyzer = feature_cfg.get("tfidf_analyzer", "char_wb")
        ngram_min = int(feature_cfg.get("tfidf_ngram_min", 3))
        ngram_max = int(feature_cfg.get("tfidf_ngram_max", 5))
        min_df = int(feature_cfg.get("tfidf_min_df", 1))
        max_features = int(feature_cfg.get("tfidf_max_features", 20000))

        self.param_kv_vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_features=max_features,
        )
        self.param_key_vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_features=max_features,
        )
        self.path_vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_features=max_features,
        )
        self.http_vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_features=max_features,
        )
        self.stats_scaler = RobustScaler()
        self.template_parser = UrlTemplateParser()

    def fit_transform(self, logs: List[Dict]):
        texts_param_kv, texts_param_key, texts_path, texts_http, texts_full_url, stats = self._prepare_parts(logs)
        x_param_kv = self._safe_fit_transform(self.param_kv_vectorizer, texts_param_kv)
        x_param_key = self._safe_fit_transform(self.param_key_vectorizer, texts_param_key)
        x_path = self._safe_fit_transform(self.path_vectorizer, texts_path)
        x_http = self._safe_fit_transform(self.http_vectorizer, texts_http)
        x_stats = csr_matrix(self.stats_scaler.fit_transform(stats))
        return hstack([x_http, x_param_kv, x_param_key, x_path, x_stats], format="csr")

    def transform(self, logs: List[Dict]):
        texts_param_kv, texts_param_key, texts_path, texts_http, texts_full_url, stats = self._prepare_parts(logs)
        x_param_kv = self._safe_transform(self.param_kv_vectorizer, texts_param_kv)
        x_param_key = self._safe_transform(self.param_key_vectorizer, texts_param_key)
        x_path = self._safe_transform(self.path_vectorizer, texts_path)
        x_http = self._safe_transform(self.http_vectorizer, texts_http)
        x_stats = csr_matrix(self.stats_scaler.transform(stats))
        return hstack([x_http, x_param_kv, x_param_key, x_path, x_stats], format="csr")

    def _prepare_parts(self, logs: List[Dict]) -> Tuple[List[str], List[str], List[str], List[str], List[str], np.ndarray]:
        texts_param_kv: List[str] = []
        texts_param_key: List[str] = []
        texts_path: List[str] = []
        texts_http: List[str] = []
        texts_full_url: List[str] = []
        stats_rows: List[List[float]] = []

        for rec in logs:
            # 1. 解析URL（从http请求行或uri字段）
            method, path_raw, query_raw = parse_url_from_log(rec)
            
            # 2. 模板化URL（路径 + 查询参数一起模板化）
            full_url_raw = f"{path_raw}?{query_raw}" if query_raw else path_raw
            full_url_templated = self.template_parser.parse(full_url_raw)
            
            # 3. 分离模板化后的路径和查询参数
            if "?" in full_url_templated:
                path_templated, query_templated = full_url_templated.split("?", 1)
            else:
                path_templated, query_templated = full_url_templated, ""
            
            # 4. 解析headers和body
            http_raw = rec.get("http", "")
            http_decoded = http_raw.replace("\\/", "/").replace("\\r\\n", "\r\n")
            
            headers = {}
            body = ""
            
            if "\r\n\r\n" in http_decoded:
                header_section, body_section = http_decoded.split("\r\n\r\n", 1)
                lines = header_section.split("\r\n")
                for line in lines[1:]:  # 跳过请求行
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        headers[key.lower()] = normalize_value(value)
                body = normalize_value(body_section.split("\r\n")[0] if body_section else "")
            
            # 5. 提取参数对（使用模板化后的查询参数）
            param_pairs = extract_param_pairs(method, query_templated, body, headers)
            
            # 6. 构建文本特征
            texts_param_kv.append(" ".join([f"{k}={v}" for k, v in param_pairs]))
            texts_param_key.append(" ".join([k for k, _ in param_pairs]))
            texts_path.append(path_templated)
            texts_full_url.append(full_url_templated)
            
            header_text = " ".join([f"{k}:{v}" for k, v in headers.items()])
            texts_http.append(f"{method} {path_templated} {query_templated} {header_text} {body}".strip())
            
            # 7. 手工特征（使用模板化后的值）
            stats = extract_enhanced_features(
                path=path_templated,
                query=query_templated,
                headers=headers,
                body=body,
                method=method,
                param_pairs=param_pairs
            )
            stats_rows.append(stats)

        return texts_param_kv, texts_param_key, texts_path, texts_http, texts_full_url, np.asarray(stats_rows, dtype=float)

    @staticmethod
    def _safe_fit_transform(vectorizer: TfidfVectorizer, texts: List[str]):
        non_empty = any((x or "").strip() for x in texts)
        if not non_empty:
            return csr_matrix((len(texts), 0), dtype=float)
        return vectorizer.fit_transform(texts)

    @staticmethod
    def _safe_transform(vectorizer: TfidfVectorizer, texts: List[str]):
        if not hasattr(vectorizer, "vocabulary_") or len(getattr(vectorizer, "vocabulary_", {})) == 0:
            return csr_matrix((len(texts), 0), dtype=float)
        return vectorizer.transform(texts)