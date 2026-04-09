from typing import Dict, List

from src.utils.io_utils import load_json


def _normalize_record(record: Dict) -> Dict:
    return {
        "uri": record.get("uri", ""),
        "method": record.get("method", ""),
        "alertlevel": record.get("alertlevel", ""),
        "event_type": record.get("event_type", ""),
        "http": record.get("http", ""),
        "raw_client_ip": record.get("raw_client_ip", ""),
        "stat_time": record.get("stat_time", ""),
    }


def load_normal_logs(get_path: str, post_path: str) -> Dict[str, List[Dict]]:
    get_logs = [_normalize_record(x) for x in load_json(get_path)]
    post_logs = [_normalize_record(x) for x in load_json(post_path)]
    return {"GET": get_logs, "POST": post_logs}
