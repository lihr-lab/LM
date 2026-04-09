#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单元测试：验证 POST 增强特征提取器能否正确处理包含 WAF 告警附加信息的 JSON 请求
"""

import json
import sys
import os

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.features import (
    parse_url_from_log,
    extract_post_enhanced_features,
    FeatureBuilder
)


# ==================== 测试数据 ====================
# 从你提供的攻击日志中提取的原始记录（保持转义字符）
ATTACK_LOG = {
    "uri": "\\/jira\\/rest\\/analytics\\/1.0\\/publish\\/bulk",
    "method": "POST",
    "alertlevel": "HIGH",
    "event_type": "SQL_Injection",
    "http": "POST \\/jira\\/rest\\/analytics\\/1.0\\/publish\\/bulk HTTP\\/1.1\\r\\nHost: inone.intra.nsfocus.com\\r\\nConnection: keep-alive\\r\\nContent-Length: 4636\\r\\nCache-Control: max-age=0\\r\\nsec-ch-ua: \\\"Google Chrome\\\";v=\\\"107\\\", \\\"Chromium\\\";v=\\\"107\\\", \\\"Not=A?Brand\\\";v=\\\"24\\\"\\r\\nsec-ch-ua-platform: \\\"Windows\\\"\\r\\nsec-ch-ua-mobile: ?0\\r\\nUser-Agent: Mozilla\\/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit\\/537.36 (KHTML, like Gecko) Chrome\\/107.0.0.0 Safari\\/537.36\\r\\nContent-Type: application\\/json\\r\\nAccept: *\\/*\\r\\nOrigin: https:\\/\\/inone.intra.nsfocus.com\\r\\nSec-Fetch-Site: same-origin\\r\\nSec-Fetch-Mode: cors\\r\\nSec-Fetch-Dest: empty\\r\\nReferer: https:\\/\\/inone.intra.nsfocus.com\\/jira\\/issues\\/?filter=-2\\r\\nAccept-Encoding: gzip, deflate, br\\r\\nAccept-Language: zh-CN,zh;q=0.9\\r\\nCookie: ngx_jira=192.168.142.202:8080; seraph.rememberme.cookie=2252840%3A4652cb2ce535bf9ee6151b6b1469ffc746cf114d; JSESSIONID=B3E102AF4E986B906874C93E262982C1; atlassian.xsrf.token=BW0V-G7S3-B591-HO9N_31fa6674da12d4aaa215c7b1d3f695d6034cb70e_lin; mywork.tab.tasks=false\\r\\n\\r\\n[{\\\"name\\\":\\\"kickass.criteriaAutoUpdateEnabled\\\",\\\"properties\\\":{\\\"enabled\\\":false},\\\"timeDelta\\\":-7109},{\\\"name\\\":\\\"kickass.issue-navigator.change-layout\\\",\\\"properties\\\":{\\\"type\\\":\\\"list-view\\\"},\\\"timeDelta\\\":-7069},{\\\"name\\\":\\\"quicksearch.enabled\\\",\\\"properties\\\":{},\\\"timeDelta\\\":-6847},{\\\"name\\\":\\\"wrm.caching.data.collector\\\",\\\"properties\\\":{\\\"SSL\\\":true,\\\"AssetsForeignOrigin\\\":false,\\\"CacheHits\\\":22,\\\"CacheHitSize\\\":9311139,\\\"CacheMisses\\\":0,\\\"CacheMissedSize\\\":0,\\\"CacheHitsJs\\\":14,\\\"CacheHitSizeJs\\\":8881495,\\\"CacheMissesJs\\\":0,\\\"CacheMissedSizeJs\\\":0,\\\"CacheHitsCss\\\":8,\\\"CacheHitSizeCss\\\":429644,\\\"CacheMissesCss\\\":0,\\\"CacheMissedSizeCss\\\":0},\\\"timeDelta\\\":-6580},{\\\"name\\\":\\\"browser.metrics.navigation\\\",\\\"properties\\\":{\\\"ttfb\\\":854.8999999761581,\\\"pageVisibility\\\":\\\"hidden\\\",\\\"key\\\":\\\"jira.issue.nav-list\\\",\\\"isInitial\\\":\\\"true\\\",\\\"threshold\\\":\\\"1000\\\",\\\"userDeviceMemory\\\":8,\\\"userDeviceProcessors\\\":28,\\\"apdex\\\":\\\"0.5\\\",\\\"firstPaint\\\":\\\"917\\\",\\\"journeyId\\\":\\\"4ec75dec-d9d1-4561-8ec4-d437e6e5d584\\\",\\\"navigationType\\\":\\\"0\\\",\\\"readyForUser\\\":\\\"1473\\\",\\\"redirectCount\\\":\\\"0\\\",\\\"resourceLoadedEnd\\\":\\\"1097\\\",\\\"resourceLoadedStart\\\":868.1999999284744,\\\"unloadEventStart\\\":\\\"860\\\",\\\"unloadEventEnd\\\":\\\"861\\\",\\\"fetchStart\\\":\\\"12\\\",\\\"domainLookupStart\\\":\\\"12\\\",\\\"domainLookupEnd\\\":\\\"12\\\",\\\"connectStart\\\":\\\"12\\\",\\\"connectEnd\\\":\\\"12\\\",\\\"requestStart\\\":\\\"13\\\",\\\"responseStart\\\":\\\"855\\\",\\\"responseEnd\\\":\\\"955\\\",\\\"domLoading\\\":\\\"866\\\",\\\"domInteractive\\\":\\\"975\\\",\\\"domContentLoadedEventStart\\\":\\\"1478\\\",\\\"domContentLoadedEventEnd\\\":\\\"1478\\\",\\\"domComplete\\\":\\\"1599\\\",\\\"loadEventStart\\\":\\\"1599\\\",\\\"loadEventEnd\\\":\\\"1600\\\",\\\"userAgent\\\":\\\"Mozilla\\/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit\\/537.36 (KHTML, like Gecko) Chrome\\/107.0.0.0 Safari\\/537.36\\\",\\\"deferScriptsClicks\\\":0,\\\"deferScriptsKeydowns\\\":0,\\\"correlationId\\\":\\\"6aed29d1460fb0\\\",\\\"effectiveType\\\":\\\"4g\\\",\\\"downlink\\\":10,\\\"rtt\\\":0,\\\"serverDuration\\\":\\\"837\\\",\\\"dbReadsTimeInMs\\\":\\\"41\\\",\\\"dbConnsTimeInMs\\\":\\\"124\\\",\\\"applicationHash\\\":\\\"a11467c076dd81329ea6f81fc78c3dc81a1a40aa\\\",\\\"elementTimings\\\":\\\"{\\\\\\\"app-header\\\\\\\":931.0999999046326}\\\",\\\"resourceTiming\\\":\\\"{\\\\\\\"b\\/\\\\\\\":{\\\\\\\"jira.\\\\\\\":{\\\\\\\"filter.deletion.warning:styles\\/<.css\\\\\\\":[\\\\\\\"2,o4,o5,o5,o4,o4,o4,o4,o4,o4\\\\\\\"],\\\\\\\"webrs:calendar-\\\\\\\":{\\\\\\\"en\\/<.js\\\\\\\":[\\\\\\\"3,o5,o5,,,o5,o5,o5,o5,o5\\\\\\\"],\\\\\\\"localisation-moment\\/<.js\\\\\\\":[\\\\\\\"3,o5,o5,,,o5,o5,o5,o5,o5\\\\\\\"]}},\\\\\\\"com.\\\\\\\":{\\\\\\\"a.jira.jira-tzdetect-plugin:tzdetect-\\\\\\\":{\\\\\\\"banner-component\\/<.\\\\\\\":{\\\\\\\"css\\\\\\\":[\\\\\\\"2,o4,ob,o5,o5,o4,o4,o4,o4,o4\\\\\\\"],\\\\\\\"js\\\\\\\":[\\\\\\\"3,o5,oj,oj,o9,o5,o5,o5,o5,o5\\\\\\\"]},\\\\\\\"lib\\/<.js\\\\\\\":[\\\\\\\"3,o5,oj,oj,o9,o5,o5,o5,o5,o5\\\\\\\"]},\\\\\\\"go2group.jira.plugin.synapse:synapse-jquery-autocomplete-rs\\/<.js\\\\\\\":[\\\\\\\"3,o5,o5,,,o5,o5,o5,o5,o5\\\\\\\"]}},\\\\\\\"contextb\\/\\\\\\\":{\\\\\\\"css\\/\\\\\\\":{\\\\\\\"_super,-flush-app-header-early-inline-rs,-com.a.plugins.a.plugins-webr-rest:data-collector-perf-observer,-jira.filter.deletion.warning:styles,-jira.webrs:r-phase-checkpoint-init\\/batch.css\\\\\\\":[\\\\\\\"2,o4,oi,o5,o4,o4,o4,o4,o4,o4\\\\\\\"],\\\\\\\"jira.\\\\\\\":{\\\\\\\"view.issue,atl.general,viewissue.standalone,jira.navigator,jira.navigator.kickass,jira.general,jira.global,jira.navigator.simple,jira.navigator.advanced,-_super\\/batch.css\\\\\\\":[\\\\\\\"2,o4,p0,o5,o4,o4,o4,o4,o4,o4\\\\\\\"],\\\\\\\"global.look-and-feel,-_super\\/batch.css\\\\\\\":[\\\\\\\"2,o4,ob,o5,o4,o4,o4,o4,o4,o4\\\\\\\"]}},\\\\\\\"js\\/\\\\\\\":{\\\\\\\"_super,-flush-app-header-early-inline-rs,-com.a.plugins.a.plugins-webr-rest:data-collector-perf-observer,-jira.filter.deletion.wa;alertinfo:sql attack found in content_parameter!;proxy_info:;charaters:;count_num:1;protocol_type:HTTPS;wci:zqb2s37arWzQoLXGytgXtc9Hv7NaZfj0viy7Cw==;wsi:jKPR8W9pJn24aW4p908FoLd88bgAAAAAy0ESlA==;country:LN;correlation_id:7618846167215850292;site_name:jira;vsite_name:None;raw_client_ip:10.67.6.155;proxy_port:443;ser_status_code:200;is_api:NO;api_event_type:0\"\"}",
    "raw_client_ip": "10.67.6.155",
    "stat_time": "2026-03-19 14:14:18"
}


def test_parse_url_from_log():
    """测试 URL 解析函数"""
    print("\n[TEST] parse_url_from_log")
    method, path, query = parse_url_from_log(ATTACK_LOG)
    print(f"  method: {method}")
    print(f"  path: {path}")
    print(f"  query: {query}")

    assert method == "POST"
    assert "/jira/rest/analytics/1.0/publish/bulk" in path
    print("  ✅ PASS")


def test_extract_post_enhanced_features():
    """测试 POST 增强特征提取（核心）"""
    print("\n[TEST] extract_post_enhanced_features")

    # 模拟从日志中解析 headers 和 body（简化版，直接使用原始 http 解码后提取）
    http_raw = ATTACK_LOG["http"]
    http_decoded = http_raw.replace("\\/", "/").replace("\\r\\n", "\r\n")

    # 提取 headers 和 body（与 FeatureBuilder._prepare_parts 逻辑一致）
    headers = {}
    body = ""
    if "\r\n\r\n" in http_decoded:
        header_section, body_section = http_decoded.split("\r\n\r\n", 1)
        lines = header_section.split("\r\n")
        for line in lines[1:]:
            if ": " in line:
                key, value = line.split(": ", 1)
                headers[key.lower()] = value
        # 注意：body_section 可能包含 JSON + 附加信息，需要只取第一部分（直到遇到 ; 或换行）
        # 实际 FeatureBuilder 中使用了 .split("\r\n")[0] 来只取第一行，这里模拟
        body = body_section.split("\r\n")[0] if body_section else ""

    print(f"  Content-Type: {headers.get('content-type', '')}")
    print(f"  Body (前200字符): {body[:200]}...")
    body_start = body[0] if body else "EMPTY"
    print(f"  Body 开头: {body_start}")

    # 调用被测试函数
    json_stats, json_paths_text, body_text, param_keys_text = extract_post_enhanced_features(body, headers)

    print(f"\n  JSON 统计特征 (8维): {json_stats}")
    print(f"  JSON 路径数量: {len(json_paths_text.split()) if json_paths_text else 0}")
    if json_paths_text:
        print(f"  JSON 路径（50字符）: {json_paths_text[:50]}...")
    else:
        print("  JSON 路径：(空)")
    print(f"  Body 文本长度: {len(body_text)}")
    print(f"  参数键文本: {param_keys_text[:200] if param_keys_text else '(空)'}")

    # 断言
    # 1. body 应该有内容
    assert len(body) > 0, "Body 不能为空"

    # 2. JSON 统计特征中应该有有效数据
    assert json_stats[0] > 0 or json_stats[1] > 0, "应该提取到 JSON 结构信息"

    print("  ✅ PASS")


def test_full_feature_builder():
    """测试完整的 FeatureBuilder 流程（包括 TF-IDF 拟合）"""
    print("\n[TEST] Full FeatureBuilder (fit_transform + transform)")

    from src.features import FeatureBuilder

    # 创建特征构建器（使用最小配置，避免内存过大）
    feature_cfg = {
        "tfidf_analyzer": "char_wb",
        "tfidf_ngram_min": 2,
        "tfidf_ngram_max": 4,
        "tfidf_min_df": 1,
        "tfidf_max_features": 1000,  # 小词汇表加速测试
    }
    builder = FeatureBuilder(feature_cfg)

    # 模拟日志数据
    logs = [ATTACK_LOG]

    # 训练（拟合）
    X_train = builder.fit_transform(logs)
    print(f"  训练特征矩阵形状: {X_train.shape}")
    assert X_train.shape[0] == 1, "样本数应该是1"
    assert X_train.shape[1] > 0, "特征维度应该 > 0"

    # 预测（转换）
    X_test = builder.transform(logs)
    print(f"  预测特征矩阵形状: {X_test.shape}")
    assert X_test.shape == X_train.shape, "训练和预测的特征维度应该一致"

    # 检查一些向量化器状态
    assert hasattr(builder, 'post_json_path_vectorizer'), "应该有post_json_path_vectorizer"
    assert hasattr(builder, 'post_body_vectorizer'), "应该有post_body_vectorizer"

    print("  ✅ PASS")


def main():
    print("="*60)
    print("POST 增强特征提取单元测试")
    print("="*60)
    
    try:
        test_parse_url_from_log()
        test_extract_post_enhanced_features()
        test_full_feature_builder()
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 发生异常: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()