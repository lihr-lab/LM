# read_waf.py - 筛选Jira站点的正常日志（前1000条，按GET/POST分别存储）
import re
import json

file_path = "D:\\browser\\waf_10.67.10.72.json-20260321"
normal_output_file_get = "waf_classified_normal_get_1.json"
normal_output_file_post = "waf_classified_normal_post_1.json"

NORMAL_TAG = 'waf_log_webaccess'

# 需要保留的字段
KEEP_FIELDS = ['uri', 'method', 'alertlevel', 'event_type', 'http', 'raw_client_ip', 'stat_time']

def is_jira_log(line):
    return 'inone.intra.nsfocus.com' in line or '/jira/' in line

def extract_fields(line):
    """提取指定字段"""
    fields = {}
    
    # 提取 uri 或 url
    uri = re.search(r'uri:([^;]+?)(?=;\w+:|$)', line)
    if uri:
        fields['uri'] = uri.group(1).strip().strip('"')
    else:
        url = re.search(r'url:([^;]+?)(?=;\w+:|$)', line)
        if url:
            fields['uri'] = url.group(1).strip().strip('"')
    
    # method
    method = re.search(r'method:([^;]+?)(?=;\w+:|$)', line)
    if method:
        fields['method'] = method.group(1).strip()
    
    # alertlevel
    alertlevel = re.search(r'alertlevel:([^;]+?)(?=;\w+:|$)', line)
    if alertlevel:
        fields['alertlevel'] = alertlevel.group(1).strip()
    
    # event_type
    event_type = re.search(r'event_type:([^;]+?)(?=;\w+:|$)', line)
    if event_type:
        fields['event_type'] = event_type.group(1).strip()
    
    # http - 修复：提取到行尾或遇到 ;http_protocol: 或 ;wsi: 等
    # 先尝试匹配到 ;http_protocol:
    http_match = re.search(r'http:(.+?)(?=;http_protocol:|$)', line, re.DOTALL)
    if not http_match:
        # 如果没找到，匹配到 ;wsi: 或行尾
        http_match = re.search(r'http:(.+?)(?=;wsi:|$)', line, re.DOTALL)
    if not http_match:
        # 直接取到最后
        http_match = re.search(r'http:(.+)$', line)
    
    if http_match:
        http_val = http_match.group(1).strip()
        # 去掉末尾可能残留的引号
        if http_val.endswith('"'):
            http_val = http_val[:-1]
        # 还原换行符
        http_val = http_val.replace('#015#012', '\\r\\n')
        fields['http'] = http_val
    
    # raw_client_ip
    raw_client_ip = re.search(r'raw_client_ip:([^;]+?)(?=;\w+:|$)', line)
    if raw_client_ip:
        fields['raw_client_ip'] = raw_client_ip.group(1).strip()
    
    # stat_time
    stat_time = re.search(r'stat_time:([^;]+?)(?=;\w+:|$)', line)
    if stat_time:
        fields['stat_time'] = stat_time.group(1).strip()
    
    return fields

print("=" * 60)
print("开始筛选Jira正常访问日志（按GET/POST分别存储）...")
print(f"输出字段: {KEEP_FIELDS}")
print("=" * 60)

normal_logs_get = []   # GET 请求存储
normal_logs_post = []  # POST 请求存储
scan_count = 0
TARGET = 2000  # 每种类型最多收集2000条

with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    for line_num, line in enumerate(f, 1):
        # 检查是否已达到目标数量（两种类型都满）
        if len(normal_logs_get) >= TARGET and len(normal_logs_post) >= TARGET:
            print(f"\n已达到目标数量（GET: {TARGET}, POST: {TARGET}），停止扫描。")
            break
        
        line = line.strip()
        if not line:
            continue
        
        scan_count += 1
        
        if scan_count % 5000 == 0:
            print(f"  已扫描 {scan_count} 行，已找到 GET: {len(normal_logs_get)} 条, POST: {len(normal_logs_post)} 条...")
        
        # 只筛选Jira日志
        if not is_jira_log(line):
            continue
        
        # 检查是否是正常访问日志
        tag_match = re.search(r'tag:([^;\s]+)', line)
        tag = tag_match.group(1) if tag_match else None
        
        if tag != NORMAL_TAG:
            continue
        
        # 提取字段
        fields = extract_fields(line)
        
        # 只保留指定字段
        filtered_fields = {k: fields[k] for k in KEEP_FIELDS if k in fields}
        
        # 获取 method
        method = filtered_fields.get('method', '')
        
        # 根据 method 分类存储
        if method == 'GET' and len(normal_logs_get) < TARGET:
            normal_logs_get.append(filtered_fields)
        elif method == 'POST' and len(normal_logs_post) < TARGET:
            normal_logs_post.append(filtered_fields)
        # 忽略其他 method（如 PUT, DELETE 等）

print(f"\n完成！")
print(f"  总扫描行数: {scan_count}")
print(f"  找到的 GET 请求: {len(normal_logs_get)} 条")
print(f"  找到的 POST 请求: {len(normal_logs_post)} 条")

# 保存 GET 请求
with open(normal_output_file_get, 'w', encoding='utf-8') as f:
    json.dump(normal_logs_get, f, ensure_ascii=False, indent=2)

# 保存 POST 请求
with open(normal_output_file_post, 'w', encoding='utf-8') as f:
    json.dump(normal_logs_post, f, ensure_ascii=False, indent=2)

print(f"\nGET 请求已保存到: {normal_output_file_get}")
print(f"POST 请求已保存到: {normal_output_file_post}")

# 验证
print("\n" + "=" * 60)
print("验证结果:")
if normal_logs_get:
    first_get = normal_logs_get[0]
    print(f"GET 请求示例 - 包含的字段: {list(first_get.keys())}")
    if 'http' in first_get:
        print(f"  http字段长度: {len(first_get['http'])} 字符")
        print(f"  http字段前150字符: {first_get['http'][:150]}...")

if normal_logs_post:
    first_post = normal_logs_post[0]
    print(f"\nPOST 请求示例 - 包含的字段: {list(first_post.keys())}")
    if 'http' in first_post:
        print(f"  http字段长度: {len(first_post['http'])} 字符")
        print(f"  http字段前150字符: {first_post['http'][:150]}...")

print("\n完成！")