"""Validate detector telemetry; an old/missing frame is never a zero count."""
from datetime import datetime, timezone

MAX_SIGNAL_AGE_SECONDS = 30


def select_observation(metrics, sources, source_id=None, now=None):
    now = now or datetime.now(timezone.utc)
    if not metrics:
        raise ValueError('尚無人物訊號，請先啟動影片或攝影機偵測。')
    streams = metrics.get('streams') or {}
    if source_id:
        metric = streams.get(source_id) if streams else (metrics if source_id == (metrics.get('stream_id') or 'default') else None)
        if metric is None:
            raise ValueError('找不到指定的偵測來源，請核對來源代號。')
    elif streams:
        if len(streams) != 1:
            raise ValueError('存在多個偵測來源，請指定來源代號，不能混合人數。')
        source_id, metric = next(iter(streams.items()))
    else:
        metric = metrics
        source_id = metric.get('stream_id') or 'default'
    source = next((item for item in sources if item.get('source_id') == source_id), None)
    if source and source.get('status') != 'online':
        raise ValueError('偵測已停止或來源已離線；這不代表畫面無人。')
    try:
        observed_at = datetime.fromisoformat(metric['timestamp'].replace('Z', '+00:00'))
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        age = (now - observed_at).total_seconds()
        count = metric['people_count']
        if isinstance(count, bool) or not isinstance(count, (int, float)) or not 0 <= count <= 50 or int(count) != count:
            raise ValueError()
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError('人物訊號格式不完整，不能將它當作零人。') from None
    if age < -5 or age > MAX_SIGNAL_AGE_SECONDS:
        raise ValueError('人物訊號已過期或時間異常；請確認偵測仍在執行。')
    return int(count), observed_at, source_id

