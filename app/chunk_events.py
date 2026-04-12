import json
import queue
import threading
import time
import uuid


class ChunkEventBroker:
    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers = {}

    def subscribe(self, *, chapter=None, scope_mode="chapter"):
        subscriber_id = uuid.uuid4().hex
        subscriber_queue = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers[subscriber_id] = {
                "queue": subscriber_queue,
                "chapter": str(chapter or "").strip(),
                "scope_mode": str(scope_mode or "chapter").strip().lower(),
            }
        return subscriber_id, subscriber_queue

    def unsubscribe(self, subscriber_id):
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def publish(self, event_type, data):
        payload = {
            "type": str(event_type or "").strip(),
            "data": data,
            "at": time.time(),
        }
        encoded = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            subscribers = list(self._subscribers.items())

        for subscriber_id, subscriber in subscribers:
            if not self._should_deliver(subscriber, payload):
                continue
            q = subscriber["queue"]
            try:
                q.put_nowait(encoded)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(encoded)
                except queue.Full:
                    self.unsubscribe(subscriber_id)

    @staticmethod
    def _should_deliver(subscriber, payload):
        scope_mode = subscriber.get("scope_mode") or "chapter"
        chapter = subscriber.get("chapter") or ""
        event_type = payload.get("type")
        data = payload.get("data") or {}

        if scope_mode != "chapter" or not chapter:
            return True

        if event_type in {"audio_status", "chapter_list_changed"}:
            return True
        if event_type == "chapter_deleted":
            return str(data.get("chapter") or "").strip() == chapter
        if event_type in {"chunk_upsert", "chunk_delete"}:
            return str(data.get("chapter") or "").strip() == chapter
        return True


chunk_event_broker = ChunkEventBroker()

