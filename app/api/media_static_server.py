"""Dedicated static media server used by the voicelines proxy.

This subprocess keeps the proxy architecture intact while serving project files
with Starlette's StaticFiles implementation, which supports browser audio
requirements such as byte-range requests.
"""

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from runtime_layout import LAYOUT

BASE_DIR = LAYOUT.app_dir
ROOT_DIR = LAYOUT.project_dir
VOICELINES_DIR = LAYOUT.voicelines_dir

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/__health")
async def health():
    return {"status": "ok"}


app.mount("/voicelines", StaticFiles(directory=VOICELINES_DIR), name="voicelines")
