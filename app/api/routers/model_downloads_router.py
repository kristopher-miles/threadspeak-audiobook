import asyncio
import inspect
import json
import queue as stdlib_queue

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from model_downloads import model_download_manager


router = APIRouter()


def _sse_frame(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.get("/api/model_downloads/status")
async def model_downloads_status():
    return model_download_manager.snapshot()


@router.get("/api/model_downloads/events")
async def model_downloads_events():
    subscriber_id, subscriber_queue = model_download_manager.subscribe()

    async def stream():
        try:
            yield _sse_frame("snapshot", model_download_manager.snapshot())
            while True:
                try:
                    if inspect.iscoroutinefunction(getattr(subscriber_queue, "get", None)):
                        payload = await subscriber_queue.get()
                    else:
                        payload = await asyncio.to_thread(subscriber_queue.get, True, 15.0)
                    yield _sse_frame("snapshot", payload or {"downloads": []})
                except stdlib_queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            model_download_manager.unsubscribe(subscriber_id)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/api/model_downloads/retry/{download_id}")
async def retry_model_download(download_id: str):
    try:
        return model_download_manager.retry_download(download_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Download not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
