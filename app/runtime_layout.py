import os
import tempfile
from dataclasses import dataclass


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


@dataclass(frozen=True)
class RuntimeLayout:
    app_dir: str
    repo_root: str
    config_dir: str
    prompts_dir: str
    runtime_dir: str
    current_runtime_dir: str
    project_dir: str
    runs_dir: str
    resources_dir: str
    builtin_lora_dir: str

    voices_path: str
    voice_config_path: str
    script_path: str
    state_path: str
    chunks_path: str
    paragraphs_path: str

    workflow_dir: str
    processing_workflow_state_path: str
    new_mode_workflow_state_path: str
    audio_queue_state_path: str
    audio_cancel_tombstone_path: str
    script_generation_checkpoint_path: str
    script_review_checkpoint_path: str

    db_dir: str
    chunks_db_path: str
    chunks_queue_log_path: str
    transcription_cache_path: str
    voice_audit_log_path: str

    repair_dir: str
    script_sanity_path: str
    script_repair_trace_path: str

    exports_dir: str
    audiobook_path: str
    optimized_export_path: str
    audacity_export_path: str
    m4b_path: str
    m4b_cover_path: str
    sanity_trim_first_clip_path: str
    sanity_assemble_first5_path: str
    sanity_assemble_first5_normalized_path: str

    uploads_dir: str
    voicelines_dir: str
    clone_voices_dir: str
    designed_voices_dir: str

    archives_dir: str
    script_snapshots_dir: str
    project_archives_dir: str
    backups_dir: str
    chunk_backups_dir: str

    workspace_dir: str
    dataset_builder_dir: str
    lora_datasets_dir: str
    lora_models_dir: str

    logs_dir: str
    llm_responses_log_path: str
    review_responses_log_path: str

    prompt_default_path: str
    prompt_review_path: str
    prompt_attribution_path: str
    prompt_voice_path: str
    prompt_dialogue_path: str
    prompt_temperament_path: str

    legacy_root_dir: str

    @classmethod
    def from_app_dir(cls, app_dir: str) -> "RuntimeLayout":
        app_dir = os.path.abspath(app_dir)
        repo_root = os.path.dirname(app_dir)
        config_dir = os.path.join(repo_root, "config")
        prompts_dir = os.path.join(config_dir, "prompts")
        runtime_dir = os.path.join(repo_root, "runtime")
        current_runtime_dir = os.path.join(runtime_dir, "current")
        project_dir = os.path.join(current_runtime_dir, "project")
        runs_dir = os.path.join(runtime_dir, "runs")
        resources_dir = os.path.join(app_dir, "resources")
        builtin_lora_dir = os.path.join(resources_dir, "builtin_lora")

        workflow_dir = os.path.join(project_dir, "workflow")
        db_dir = os.path.join(project_dir, "db")
        repair_dir = os.path.join(project_dir, "repair")
        exports_dir = os.path.join(project_dir, "exports")
        uploads_dir = os.path.join(project_dir, "uploads")
        voicelines_dir = os.path.join(project_dir, "voicelines")
        clone_voices_dir = os.path.join(project_dir, "clone_voices")
        designed_voices_dir = os.path.join(project_dir, "designed_voices")
        archives_dir = os.path.join(project_dir, "archives")
        script_snapshots_dir = os.path.join(archives_dir, "script_snapshots")
        project_archives_dir = os.path.join(archives_dir, "project_archives")
        backups_dir = os.path.join(archives_dir, "backups")
        chunk_backups_dir = os.path.join(backups_dir, "chunks")
        workspace_dir = os.path.join(project_dir, "workspace")
        dataset_builder_dir = os.path.join(workspace_dir, "dataset_builder")
        lora_datasets_dir = os.path.join(workspace_dir, "lora_datasets")
        lora_models_dir = os.path.join(workspace_dir, "lora_models")
        logs_dir = os.path.join(project_dir, "logs")

        return cls(
            app_dir=app_dir,
            repo_root=repo_root,
            config_dir=config_dir,
            prompts_dir=prompts_dir,
            runtime_dir=runtime_dir,
            current_runtime_dir=current_runtime_dir,
            project_dir=project_dir,
            runs_dir=runs_dir,
            resources_dir=resources_dir,
            builtin_lora_dir=builtin_lora_dir,
            voices_path=os.path.join(project_dir, "voices.json"),
            voice_config_path=os.path.join(project_dir, "voice_config.json"),
            script_path=os.path.join(project_dir, "annotated_script.json"),
            state_path=os.path.join(project_dir, "state.json"),
            chunks_path=os.path.join(project_dir, "chunks.json"),
            paragraphs_path=os.path.join(project_dir, "paragraphs.json"),
            workflow_dir=workflow_dir,
            processing_workflow_state_path=os.path.join(workflow_dir, "processing_workflow_state.json"),
            new_mode_workflow_state_path=os.path.join(workflow_dir, "new_mode_workflow_state.json"),
            audio_queue_state_path=os.path.join(workflow_dir, "audio_queue_state.json"),
            audio_cancel_tombstone_path=os.path.join(workflow_dir, "audio_cancel_tombstone.json"),
            script_generation_checkpoint_path=os.path.join(workflow_dir, "script_generation_checkpoint.json"),
            script_review_checkpoint_path=os.path.join(workflow_dir, "script_review_checkpoint.json"),
            db_dir=db_dir,
            chunks_db_path=os.path.join(db_dir, "chunks.sqlite3"),
            chunks_queue_log_path=os.path.join(db_dir, "chunks.queue.log"),
            transcription_cache_path=os.path.join(db_dir, "transcription_cache.json"),
            voice_audit_log_path=os.path.join(db_dir, "voice_state.audit.jsonl"),
            repair_dir=repair_dir,
            script_sanity_path=os.path.join(repair_dir, "script_sanity_check.json"),
            script_repair_trace_path=os.path.join(repair_dir, "script_repair_trace.jsonl"),
            exports_dir=exports_dir,
            audiobook_path=os.path.join(exports_dir, "cloned_audiobook.mp3"),
            optimized_export_path=os.path.join(exports_dir, "optimized_audiobook.zip"),
            audacity_export_path=os.path.join(exports_dir, "audacity_export.zip"),
            m4b_path=os.path.join(exports_dir, "audiobook.m4b"),
            m4b_cover_path=os.path.join(exports_dir, "m4b_cover.jpg"),
            sanity_trim_first_clip_path=os.path.join(exports_dir, "trim_sanity_first_clip.wav"),
            sanity_assemble_first5_path=os.path.join(exports_dir, "assemble_sanity_first5.wav"),
            sanity_assemble_first5_normalized_path=os.path.join(exports_dir, "assemble_sanity_first5_normalized.wav"),
            uploads_dir=uploads_dir,
            voicelines_dir=voicelines_dir,
            clone_voices_dir=clone_voices_dir,
            designed_voices_dir=designed_voices_dir,
            archives_dir=archives_dir,
            script_snapshots_dir=script_snapshots_dir,
            project_archives_dir=project_archives_dir,
            backups_dir=backups_dir,
            chunk_backups_dir=chunk_backups_dir,
            workspace_dir=workspace_dir,
            dataset_builder_dir=dataset_builder_dir,
            lora_datasets_dir=lora_datasets_dir,
            lora_models_dir=lora_models_dir,
            logs_dir=logs_dir,
            llm_responses_log_path=os.path.join(logs_dir, "llm_responses.log"),
            review_responses_log_path=os.path.join(logs_dir, "review_responses.log"),
            prompt_default_path=os.path.join(prompts_dir, "default_prompts.txt"),
            prompt_review_path=os.path.join(prompts_dir, "review_prompts.txt"),
            prompt_attribution_path=os.path.join(prompts_dir, "attribution_prompts.txt"),
            prompt_voice_path=os.path.join(prompts_dir, "voice_prompt.txt"),
            prompt_dialogue_path=os.path.join(prompts_dir, "dialogue_identification_system_prompt.txt"),
            prompt_temperament_path=os.path.join(prompts_dir, "temperament_extraction_system_prompt.txt"),
            legacy_root_dir=repo_root,
        )

    def ensure_base_dirs(self) -> None:
        for path in (
            self.config_dir,
            self.prompts_dir,
            self.runtime_dir,
            self.current_runtime_dir,
            self.project_dir,
            self.runs_dir,
            self.resources_dir,
            self.builtin_lora_dir,
            self.workflow_dir,
            self.db_dir,
            self.repair_dir,
            self.exports_dir,
            self.uploads_dir,
            self.voicelines_dir,
            self.clone_voices_dir,
            self.designed_voices_dir,
            self.archives_dir,
            self.script_snapshots_dir,
            self.project_archives_dir,
            self.backups_dir,
            self.chunk_backups_dir,
            self.workspace_dir,
            self.dataset_builder_dir,
            self.lora_datasets_dir,
            self.lora_models_dir,
            self.logs_dir,
        ):
            _ensure_dir(path)

    def run_dir(self, run_id: str) -> str:
        run_id = str(run_id or "").strip() or "manual"
        return _ensure_dir(os.path.join(self.runs_dir, run_id))

    def run_subdir(self, run_id: str, name: str) -> str:
        return _ensure_dir(os.path.join(self.run_dir(run_id), name))

    def run_temp_dir(self, run_id: str) -> str:
        return self.run_subdir(run_id, "tmp")

    def run_logs_dir(self, run_id: str) -> str:
        return self.run_subdir(run_id, "logs")

    def run_exports_dir(self, run_id: str) -> str:
        return self.run_subdir(run_id, "exports")

    def temp_file(self, run_id: str, filename: str, *, subdir: str = "tmp") -> str:
        return os.path.join(self.run_subdir(run_id, subdir), filename)

    def make_named_temp_dir(self, run_id: str, prefix: str) -> str:
        return tempfile.mkdtemp(prefix=prefix, dir=self.run_temp_dir(run_id))

    def legacy_path(self, *parts: str) -> str:
        return os.path.join(self.legacy_root_dir, *parts)


LAYOUT = RuntimeLayout.from_app_dir(os.path.dirname(os.path.abspath(__file__)))
LAYOUT.ensure_base_dirs()

APP_DIR = LAYOUT.app_dir
REPO_ROOT = LAYOUT.repo_root
PROMPTS_DIR = LAYOUT.prompts_dir
RUNTIME_DIR = LAYOUT.runtime_dir
CURRENT_PROJECT_DIR = LAYOUT.project_dir
RUNS_DIR = LAYOUT.runs_dir
