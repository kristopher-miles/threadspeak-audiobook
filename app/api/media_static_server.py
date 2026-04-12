"""Dedicated static media server used by the voicelines proxy.

This subprocess keeps the proxy architecture intact while serving project files
with Starlette's StaticFiles implementation, which supports browser audio
requirements such as byte-range requests.
"""

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/__health")
async def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=ROOT_DIR), name="root")
