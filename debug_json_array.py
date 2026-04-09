#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试 JSON 数组处理
"""
import json
from src.features import _extract_json_part, extract_post_enhanced_features

# 测试 body（来自 test.py）
body_raw = (
    "[{\\\"name\\\":\\\"kickass.criteriaAutoUpdateEnabled\\\",\\\"properties\\\":{\\\"enabled\\\":false},\\\"timeDelta\\\":-7109},"
    "{\\\"name\\\":\\\"kickass.issue-navigator.change-layout\\\",\\\"properties\\\":{\\\"type\\\":\\\"list-view\\\"},\\\"timeDelta\\\":-7069}]"
)

print("=" * 80)
print("调试 JSON 数组解析")
print("=" * 80)

print("\n原始 body（前200字符）:")
print(body_raw[:200])
print(f"长度: {len(body_raw)}")

# 模拟转义处理
body_decoded = body_raw
body_decoded = body_decoded.replace('\\"', '"')
body_decoded = body_decoded.replace('\\\\', '\\')
body_decoded = body_decoded.replace('\\/', '/')
body_decoded = body_decoded.replace('\\r\\n', '').replace('\\n', '')

print("\n转义处理后（前200字符）:")
print(body_decoded[:200])

# 提取 JSON 部分
json_part = _extract_json_part(body_decoded)
print(f"\n_extract_json_part 结果（前200字符）:")
print(json_part[:200] if json_part else "(空)")

# 尝试 JSON 解析
try:
    obj = json.loads(json_part if json_part else body_decoded)
    print(f"\nJSON 解析成功！")
    print(f"  类型: {type(obj)}")
    if isinstance(obj, list):
        print(f"  数组长度: {len(obj)}")
        print(f"  第一个元素: {obj[0]}")
        if isinstance(obj[0], dict):
            print(f"  第一个元素的键: {list(obj[0].keys())}")
except json.JSONDecodeError as e:
    print(f"\nJSON 解析失败: {e}")

# 现在测试完整函数
print("\n" + "=" * 80)
print("测试 extract_post_enhanced_features")
print("=" * 80)

headers = {"content-type": "application/json"}
json_stats, json_paths_text, body_text, param_keys_text = extract_post_enhanced_features(body_raw, headers)

print(f"\nJSON 统计特征: {json_stats}")
print(f"JSON 路径文本（前200字符）: {json_paths_text[:200] if json_paths_text else '(空)'}")
print(f"Body 文本长度: {len(body_text)}")
print(f"参数键文本（前200字符）: {param_keys_text[:200] if param_keys_text else '(空)'}")

# 分析为什么路径为空
print("\n" + "=" * 80)
print("分析原因")
print("=" * 80)

if not json_paths_text:
    print("JSON 路径为空，可能原因：")

    # 尝试手动解析并检查
    try:
        test_obj = json.loads(json_part if json_part else body_decoded)
        print(f"  - JSON 解析成功，对象类型: {type(test_obj)}")
        if isinstance(test_obj, list):
            print(f"  - 是数组，长度: {len(test_obj)}")
            if len(test_obj) > 0 and isinstance(test_obj[0], dict):
                print(f"  - 第一个元素是字典，键: {list(test_obj[0].keys())}")
                print(f"  - 这些键应该被提取为路径")
    except Exception as e:
        print(f"  - JSON 解析异常: {e}")
