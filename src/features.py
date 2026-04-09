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
    PATTERNS = [
        (r'\b\d{8,}\b', '<NUM>'),
        (r'\b\d{4}-\d{2}-\d{2}\b', '<DATE>'),
        (r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '<UUID>', re.I),
        (r'\b[a-f0-9]{32}\b', '<MD5>', re.I),
        (r'\b[a-zA-Z0-9_\-]{20,}\b', '<TOKEN>'),
        (r'\b\d+(?:\.\d+)+\b', '<VERSION>'),
        (r'\b\d+\b', '<NUM>'),
    ]

    @classmethod
    def parse(cls, url: str) -> str:
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
    http_raw = rec.get("http", "")
    method = rec.get("method", "GET").upper()
    path = ""
    query = ""
    
    if http_raw:
        http_decoded = http_raw.replace("\\/", "/").replace("\\r\\n", "\r\n")
        request_line = http_decoded.split("\r\n")[0] if "\r\n" in http_decoded else http_decoded
        
        if " " in request_line and ("HTTP/" in request_line or request_line.startswith(("GET ", "POST ", "PUT ", "DELETE ", "HEAD ", "OPTIONS "))):
            parts = request_line.split(" ")
            if len(parts) >= 2:
                method = parts[0].upper()
                full_url = parts[1]
                if "?" in full_url:
                    path, query = full_url.split("?", 1)
                else:
                    path, query = full_url, ""
    
    if not path:
        uri_raw = rec.get("uri", "")
        if uri_raw:
            path = uri_raw.replace("\\/", "/")
            if http_raw and "?" in http_raw:
                match = re.search(r'\?([^\\\s"\']+)', http_raw)
                if match:
                    query = match.group(1)
    
    path = path.replace("\\/", "/")
    return method, path, query


# ==================== 3. URL模式特征提取 ====================
def extract_url_pattern_features(path: str, query: str) -> List[float]:
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

    query_params = parse_qsl(query, keep_blank_values=True) if query else []
    query_keys = [k for k, _ in query_params]
    query_values = [v for _, v in query_params]

    features.extend([
        float(len(query_params)),
        float(len(set(query_keys))),
        float(any(v.isdigit() for v in query_values)),
        float(any(v.replace('.', '').isdigit() and '.' in v for v in query_values)),
    ])

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


# ==================== 4. 增强特征提取（原有52维） ====================
def extract_enhanced_features(
    path: str,
    query: str,
    headers: Dict[str, str],
    body: str,
    method: str,
    param_pairs: List[Tuple[str, str]]
) -> List[float]:
    merged = f"{path} {query} {body}".strip()

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

    key_lengths = [len(k) for k, _ in param_pairs]
    val_lengths = [len(v) for _, v in param_pairs]

    param_features = [
        float(len(param_pairs)),
        float(len(set(k for k, _ in param_pairs))),
        float(np.mean(key_lengths)) if key_lengths else 0.0,
        float(np.std(key_lengths)) if len(key_lengths) > 1 else 0.0,
        float(np.mean(val_lengths)) if val_lengths else 0.0,
    ]

    json_features = _extract_json_features(body)

    user_agent = headers.get("user-agent", "").lower()
    merged_lower = merged.lower()

    client_features = [
        1.0 if "headless" in user_agent else 0.0,
        1.0 if any(bot in user_agent for bot in ["python-requests", "curl", "wget", "go-http-client"]) else 0.0,
        1.0 if any(plugin in merged_lower for plugin in ["gadget", "eazybi", "plugin", "rest"]) else 0.0,
    ]

    url_pattern_features = extract_url_pattern_features(path, query)

    return basic_features + param_features + json_features + client_features + url_pattern_features


def _extract_json_part(text: str) -> str:
    """从文本中提取有效的 JSON 部分（忽略后续的非 JSON 内容）。
    使用 json.JSONDecoder.raw_decode 正确处理字符串内部的括号。"""
    if not text or not text.strip():
        return ""
    text = text.strip()
    if not text.startswith(('{', '[')):
        return ""
    try:
        _, end_idx = json.JSONDecoder().raw_decode(text)
        return text[:end_idx]
    except json.JSONDecodeError:
        pass
    return text


def extract_post_enhanced_features(body: str, headers: Dict[str, str]) -> Tuple[List[float], str, str, str]:
    json_stats = [0.0] * 8
    json_paths_text = ""
    body_text = body
    param_keys_text = ""

    if not body or not body.strip():
        return json_stats, json_paths_text, body_text, param_keys_text

    # 解码转义字符
    body_decoded = body
    body_decoded = body_decoded.replace('\\"', '"')
    body_decoded = body_decoded.replace('\\\\', '\\')
    body_decoded = body_decoded.replace('\\/', '/')
    body_decoded = body_decoded.replace('\\r\\n', '').replace('\\n', '')

    # 提取有效的 JSON 部分（忽略后续的附加信息）
    json_part = _extract_json_part(body_decoded)

    try:
        obj = json.loads(json_part if json_part else body_decoded)
        paths = []
        stats = {
            "max_depth": 0, "key_count": 0, "array_count": 0,
            "max_string_len": 0, "avg_string_len": 0, "max_array_len": 0,
            "total_keys": 0, "has_nested": 0,
            "total_string_len": 0, "string_count": 0
        }
        _analyze_json_structure(obj, "", paths, stats, 0)

        # 如果 paths 仍为空（极端情况），手动添加顶层键或数组内对象的字段
        if not paths:
            if isinstance(obj, dict):
                paths = list(obj.keys())
            elif isinstance(obj, list) and len(obj) > 0:
                # 对于数组，提取所有不同的字段名（从前3个元素最多）
                field_names = set()
                for i, item in enumerate(obj[:3]):  # 最多查看前3个元素
                    if isinstance(item, dict):
                        field_names.update(item.keys())

                if field_names:
                    # 为每个字段名生成路径
                    paths = [f"[].{k}" for k in sorted(field_names)]
                else:
                    paths = [f"[{i}]" for i in range(min(3, len(obj)))]

        if stats["string_count"] > 0:
            stats["avg_string_len"] = stats["total_string_len"] / stats["string_count"]

        json_paths_text = " ".join(paths)
        json_stats = [
            float(stats["max_depth"]),
            float(stats["key_count"]),
            float(stats["array_count"]),
            float(stats["max_string_len"]),
            float(stats["avg_string_len"]),
            float(stats["max_array_len"]),
            float(stats["total_keys"]),
            float(stats["has_nested"])
        ]
    except json.JSONDecodeError:
        # 回退 form 格式
        content_type = headers.get("content-type", "").lower()
        if "x-www-form-urlencoded" in content_type or "=" in body:
            param_keys = []
            for pair in body.split("&"):
                if "=" in pair:
                    key = pair.split("=")[0]
                    param_keys.append(key)
            param_keys_text = " ".join(param_keys)
            json_stats = [
                0.0, float(len(param_keys)), 0.0, 0.0, 0.0, 0.0,
                float(len(set(param_keys))), 0.0
            ]

    return json_stats, json_paths_text, body_text, param_keys_text
def _analyze_json_structure(obj, prefix: str, paths: List[str], stats: Dict, depth: int):
    """递归分析 JSON 结构，提取路径和统计信息（改进版）"""
    if depth > stats["max_depth"]:
        stats["max_depth"] = depth

    if isinstance(obj, dict):
        stats["has_nested"] = 1
        for k, v in obj.items():
            stats["key_count"] += 1
            stats["total_keys"] += 1
            current_path = f"{prefix}.{k}" if prefix else str(k)
            paths.append(current_path)  # 记录当前路径
            # 递归处理值，深度+1
            _analyze_json_structure(v, current_path, paths, stats, depth + 1)
    elif isinstance(obj, list):
        stats["array_count"] += 1
        stats["max_array_len"] = max(stats["max_array_len"], len(obj))
        # 对数组中的每个元素进行处理
        for i, item in enumerate(obj):
            current_path = f"{prefix}[{i}]" if prefix else f"[{i}]"
            # 只对非简单类型添加路径标记
            if isinstance(item, (dict, list)):
                paths.append(current_path)
            # 递归处理元素
            _analyze_json_structure(item, current_path, paths, stats, depth + 1)
    elif isinstance(obj, str):
        stats["max_string_len"] = max(stats["max_string_len"], len(obj))
        stats["total_string_len"] += len(obj)
        stats["string_count"] += 1
    # 数字、布尔、null 等类型忽略
def _extract_json_features(body: str) -> List[float]:
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


# ==================== 6. 参数对提取 ====================
def extract_param_pairs(method: str, query: str, body: str, headers: Dict[str, str]) -> List[Tuple[str, str]]:
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


# ==================== 7. 主特征构建器 ====================
class FeatureBuilder:
    def __init__(self, feature_cfg: Dict):
        analyzer = feature_cfg.get("tfidf_analyzer", "char_wb")
        ngram_min = int(feature_cfg.get("tfidf_ngram_min", 3))
        ngram_max = int(feature_cfg.get("tfidf_ngram_max", 5))
        min_df = int(feature_cfg.get("tfidf_min_df", 1))
        max_features = int(feature_cfg.get("tfidf_max_features", 20000))

        # GET 使用的向量化器
        self.param_kv_vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=(ngram_min, ngram_max),
            min_df=min_df, max_features=max_features,
        )
        self.param_key_vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=(ngram_min, ngram_max),
            min_df=min_df, max_features=max_features,
        )
        self.path_vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=(ngram_min, ngram_max),
            min_df=min_df, max_features=max_features,
        )
        self.http_vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=(ngram_min, ngram_max),
            min_df=min_df, max_features=max_features,
        )
        
        # POST 新增的向量化器
        self.post_json_path_vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=(ngram_min, ngram_max),
            min_df=min_df, max_features=max_features,
        )
        self.post_body_vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=(ngram_min, ngram_max),
            min_df=min_df, max_features=max_features,
        )
        self.post_param_key_vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=(ngram_min, ngram_max),
            min_df=min_df, max_features=max_features,
        )
        
        self.stats_scaler = RobustScaler()
        self.post_stats_scaler = RobustScaler()
        self.template_parser = UrlTemplateParser()
        
        # 标记是否已拟合 POST 增强特征
        self.post_fitted = False

    def fit_transform(self, logs: List[Dict]):
        """训练阶段：拟合并转换特征"""
        # GET 特征
        texts_param_kv, texts_param_key, texts_path, texts_http, texts_full_url, stats = self._prepare_parts(logs)
        x_param_kv = self._safe_fit_transform(self.param_kv_vectorizer, texts_param_kv)
        x_param_key = self._safe_fit_transform(self.param_key_vectorizer, texts_param_key)
        x_path = self._safe_fit_transform(self.path_vectorizer, texts_path)
        x_http = self._safe_fit_transform(self.http_vectorizer, texts_http)
        x_stats = csr_matrix(self.stats_scaler.fit_transform(stats))
        
        # POST 增强特征（需要单独处理）
        x_post_json_path, x_post_body, x_post_param_key, post_stats = self._fit_transform_post_features(logs)
        
        # 拼接所有特征
        result = hstack([x_http, x_param_kv, x_param_key, x_path, x_stats], format="csr")
        if x_post_json_path is not None:
            result = hstack([result, x_post_json_path, x_post_body, x_post_param_key, post_stats], format="csr")
        
        self.post_fitted = True
        return result

    def transform(self, logs: List[Dict]):
        """预测阶段：转换特征"""
        texts_param_kv, texts_param_key, texts_path, texts_http, texts_full_url, stats = self._prepare_parts(logs)
        x_param_kv = self._safe_transform(self.param_kv_vectorizer, texts_param_kv)
        x_param_key = self._safe_transform(self.param_key_vectorizer, texts_param_key)
        x_path = self._safe_transform(self.path_vectorizer, texts_path)
        x_http = self._safe_transform(self.http_vectorizer, texts_http)
        x_stats = csr_matrix(self.stats_scaler.transform(stats))
        
        # POST 增强特征
        x_post_json_path, x_post_body, x_post_param_key, post_stats = self._transform_post_features(logs)
        
        result = hstack([x_http, x_param_kv, x_param_key, x_path, x_stats], format="csr")
        if x_post_json_path is not None:
            result = hstack([result, x_post_json_path, x_post_body, x_post_param_key, post_stats], format="csr")
        
        return result

    def _fit_transform_post_features(self, logs: List[Dict]):
        """提取并拟合 POST 增强特征"""
        json_paths_list = []
        body_texts_list = []
        param_keys_list = []
        post_stats_list = []
        
        for rec in logs:
            method = rec.get("method", "").upper()
            if method != "POST":
                # 非 POST 请求，填充空值
                json_paths_list.append("")
                body_texts_list.append("")
                param_keys_list.append("")
                post_stats_list.append([0.0] * 8)
                continue
            
            # 解析 body 和 headers
            http_raw = rec.get("http", "")
            http_decoded = http_raw.replace("\\/", "/").replace("\\r\\n", "\r\n")
            headers = {}
            body_raw = ""
            
            if "\r\n\r\n" in http_decoded:
                header_section, body_section = http_decoded.split("\r\n\r\n", 1)
                lines = header_section.split("\r\n")
                for line in lines[1:]:
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        headers[key.lower()] = normalize_value(value)
                body_raw = body_section.split("\r\n")[0] if body_section else ""
                if body_raw and ';alertinfo:' in body_raw:
                    body_raw = body_raw.split(';alertinfo:')[0]
            
            # 提取增强特征（使用原始 body，避免 normalize_value 破坏 JSON 中的数字）
            json_stats, json_paths_text, body_text, param_keys_text = extract_post_enhanced_features(body_raw, headers)
            
            json_paths_list.append(json_paths_text)
            body_texts_list.append(body_text)
            param_keys_list.append(param_keys_text)
            post_stats_list.append(json_stats)
        
        # 拟合向量化器
        x_json_path = self._safe_fit_transform(self.post_json_path_vectorizer, json_paths_list)
        x_body = self._safe_fit_transform(self.post_body_vectorizer, body_texts_list)
        x_param_key = self._safe_fit_transform(self.post_param_key_vectorizer, param_keys_list)
        x_stats = csr_matrix(self.post_stats_scaler.fit_transform(post_stats_list))
        
        return x_json_path, x_body, x_param_key, x_stats

    def _transform_post_features(self, logs: List[Dict]):
        """转换 POST 增强特征"""
        json_paths_list = []
        body_texts_list = []
        param_keys_list = []
        post_stats_list = []
        
        for rec in logs:
            method = rec.get("method", "").upper()
            if method != "POST":
                json_paths_list.append("")
                body_texts_list.append("")
                param_keys_list.append("")
                post_stats_list.append([0.0] * 8)
                continue
            
            http_raw = rec.get("http", "")
            http_decoded = http_raw.replace("\\/", "/").replace("\\r\\n", "\r\n")
            headers = {}
            body_raw = ""
            
            if "\r\n\r\n" in http_decoded:
                header_section, body_section = http_decoded.split("\r\n\r\n", 1)
                lines = header_section.split("\r\n")
                for line in lines[1:]:
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        headers[key.lower()] = normalize_value(value)
                body_raw = body_section.split("\r\n")[0] if body_section else ""
                if body_raw and ';alertinfo:' in body_raw:
                    body_raw = body_raw.split(';alertinfo:')[0]
            
            # 使用原始 body（不经过 normalize_value），避免数字被替换为 <NUM> 导致 JSON 解析失败
            json_stats, json_paths_text, body_text, param_keys_text = extract_post_enhanced_features(body_raw, headers)
            
            json_paths_list.append(json_paths_text)
            body_texts_list.append(body_text)
            param_keys_list.append(param_keys_text)
            post_stats_list.append(json_stats)
        
        x_json_path = self._safe_transform(self.post_json_path_vectorizer, json_paths_list)
        x_body = self._safe_transform(self.post_body_vectorizer, body_texts_list)
        x_param_key = self._safe_transform(self.post_param_key_vectorizer, param_keys_list)
        
        if hasattr(self.post_stats_scaler, "scale_"):
            x_stats = csr_matrix(self.post_stats_scaler.transform(post_stats_list))
        else:
            x_stats = csr_matrix(post_stats_list)
        
        return x_json_path, x_body, x_param_key, x_stats

    def _prepare_parts(self, logs: List[Dict]) -> Tuple[List[str], List[str], List[str], List[str], List[str], np.ndarray]:
        texts_param_kv: List[str] = []
        texts_param_key: List[str] = []
        texts_path: List[str] = []
        texts_http: List[str] = []
        texts_full_url: List[str] = []
        stats_rows: List[List[float]] = []

        for rec in logs:
            method, path_raw, query_raw = parse_url_from_log(rec)
            
            full_url_raw = f"{path_raw}?{query_raw}" if query_raw else path_raw
            full_url_templated = self.template_parser.parse(full_url_raw)
            
            if "?" in full_url_templated:
                path_templated, query_templated = full_url_templated.split("?", 1)
            else:
                path_templated, query_templated = full_url_templated, ""
            
            http_raw = rec.get("http", "")
            http_decoded = http_raw.replace("\\/", "/").replace("\\r\\n", "\r\n")
            
            headers = {}
            body = ""
            
            if "\r\n\r\n" in http_decoded:
                header_section, body_section = http_decoded.split("\r\n\r\n", 1)
                lines = header_section.split("\r\n")
                for line in lines[1:]:
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        headers[key.lower()] = normalize_value(value)
                body = normalize_value(body_section.split("\r\n")[0] if body_section else "")
            
            # GET 和 POST 共用参数对提取（保持兼容）
            param_pairs = extract_param_pairs(method, query_templated, body, headers)
            
            texts_param_kv.append(" ".join([f"{k}={v}" for k, v in param_pairs]))
            texts_param_key.append(" ".join([k for k, _ in param_pairs]))
            texts_path.append(path_templated)
            texts_full_url.append(full_url_templated)
            
            header_text = " ".join([f"{k}:{v}" for k, v in headers.items()])
            texts_http.append(f"{method} {path_templated} {query_templated} {header_text} {body}".strip())
            
            # 手工特征（原有52维）
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


# 为兼容已保存的模型，添加别名
FeatureBuilder = FeatureBuilder