from .shared import app
from .routers.config_router import router as config_router
from .routers.workflow_router import router as workflow_router
from .routers.voices_router import router as voices_router
from .routers.editor_audio_router import router as editor_audio_router
from .routers.scripts_router import router as scripts_router
from .routers.voice_designer_router import router as voice_designer_router
from .routers.clone_voices_router import router as clone_voices_router
from .routers.lora_router import router as lora_router
from .routers.dataset_builder_router import router as dataset_builder_router
from .routers.emotions_router import router as emotions_router
from .routers.model_downloads_router import router as model_downloads_router

app.include_router(config_router)
app.include_router(workflow_router)
app.include_router(voices_router)
app.include_router(editor_audio_router)
app.include_router(scripts_router)
app.include_router(voice_designer_router)
app.include_router(clone_voices_router)
app.include_router(lora_router)
app.include_router(dataset_builder_router)
app.include_router(emotions_router)
app.include_router(model_downloads_router)
