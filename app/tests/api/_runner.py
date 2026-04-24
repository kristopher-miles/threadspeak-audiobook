"""CLI runner for decomposed API tests (preserves legacy test_api.py behavior)."""

from ._common import *  # noqa: F401,F403
from .test_api_server_and_config import *  # noqa: F401,F403
from .test_api_script_and_project_state import *  # noqa: F401,F403
from .test_api_voices_and_chunks import *  # noqa: F401,F403
from .test_api_voice_design_lora_dataset import *  # noqa: F401,F403
from .test_api_generation_endpoints import *  # noqa: F401,F403

def run_all_tests():
    section("Server")
    run_test("server_reachable", test_server_reachable)

    section("Config")
    run_test("get_config", test_get_config)
    run_test("save_config_roundtrip", test_save_config_roundtrip)
    run_test("save_export_config_roundtrip", test_save_export_config_roundtrip)
    run_test("save_review_prompts_roundtrip", test_save_review_prompts_roundtrip)
    run_test("save_attribution_prompts_roundtrip", test_save_attribution_prompts_roundtrip)
    run_test("get_default_prompts", test_get_default_prompts)
    run_test("get_config_persists_missing_voice_prompt_default", test_get_config_persists_missing_voice_prompt_default)

    section("Upload")
    run_test("upload_file", test_upload_file, requires_full=True)

    section("Annotated Script")
    run_test("get_annotated_script", test_get_annotated_script)

    section("Scripts CRUD")
    run_test("save_script", test_save_script)
    run_test("list_scripts", test_list_scripts)
    run_test("load_script", test_load_script)
    run_test("delete_script", test_delete_script)
    run_test("delete_script_404", test_delete_script_404)

    section("Voices")
    run_test("get_voices", test_get_voices)
    run_test("save_voice_config", test_save_voice_config)

    section("Chunks")
    run_test("get_chunks", test_get_chunks)
    run_test("update_chunk", test_update_chunk)
    run_test("update_chunk_404", test_update_chunk_404)
    run_test("insert_chunk", test_insert_chunk)
    run_test("insert_chunk_404", test_insert_chunk_404)
    run_test("delete_chunk", test_delete_chunk)
    run_test("delete_chunk_invalid", test_delete_chunk_invalid)
    run_test("restore_chunk", test_restore_chunk)

    section("Status Polling")
    run_test("status_known_tasks", test_status_known_tasks)
    run_test("status_unknown_task", test_status_unknown_task)

    section("Voice Design")
    run_test("voice_design_list", test_voice_design_list)
    run_test("voice_design_delete_404", test_voice_design_delete_404)
    run_test("voice_design_preview", test_voice_design_preview, requires_full=True)
    run_test("voice_design_save_and_delete", test_voice_design_save_and_delete, requires_full=True)

    section("Clone Voices")
    run_test("clone_voices_list", test_clone_voices_list)
    run_test("clone_voices_upload_bad_format", test_clone_voices_upload_bad_format)
    run_test("clone_voices_delete_404", test_clone_voices_delete_404)
    run_test("clone_voices_upload_and_delete", test_clone_voices_upload_and_delete)
    run_test("clone_voices_upload_with_transcript_metadata", test_clone_voices_upload_with_transcript_metadata)
    run_test("clone_voices_upload_without_transcript_metadata", test_clone_voices_upload_without_transcript_metadata)
    run_test("clone_voices_download_with_transcript_metadata", test_clone_voices_download_with_transcript_metadata)

    section("LoRA Datasets")
    run_test("lora_list_datasets", test_lora_list_datasets)
    run_test("lora_delete_dataset_404", test_lora_delete_dataset_404)
    run_test("lora_upload_bad_file", test_lora_upload_bad_file)

    section("LoRA Models")
    run_test("lora_list_models", test_lora_list_models)
    run_test("lora_download_invalid", test_lora_download_invalid)
    run_test("lora_delete_model_404", test_lora_delete_model_404)
    run_test("lora_train_bad_dataset", test_lora_train_bad_dataset)
    run_test("lora_preview_404", test_lora_preview_404)
    run_test("lora_preview", test_lora_preview, requires_full=True)

    section("Dataset Builder")
    run_test("dataset_builder_list", test_dataset_builder_list)
    run_test("dataset_builder_create", test_dataset_builder_create)
    run_test("dataset_builder_update_meta", test_dataset_builder_update_meta)
    run_test("dataset_builder_update_rows", test_dataset_builder_update_rows)
    run_test("dataset_builder_status", test_dataset_builder_status)
    run_test("dataset_builder_cancel", test_dataset_builder_cancel)
    run_test("dataset_builder_save_no_samples", test_dataset_builder_save_no_samples)
    run_test("dataset_builder_delete", test_dataset_builder_delete)
    run_test("dataset_builder_delete_404", test_dataset_builder_delete_404)

    section("Merge / Export")
    run_test("get_audiobook", test_get_audiobook)
    run_test("get_audacity_export", test_get_audacity_export)

    section("Generation (TTS/LLM)")
    run_test("generate_script", test_generate_script, requires_full=True)
    run_test("review_script", test_review_script, requires_full=True)
    run_test("parse_voices", test_parse_voices, requires_full=True)
    run_test("generate_chunk", test_generate_chunk, requires_full=True)
    run_test("generate_batch", test_generate_batch, requires_full=True)
    run_test("generate_batch_fast", test_generate_batch_fast, requires_full=True)
    run_test("cancel_audio", test_cancel_audio)
    run_test("export_audacity", test_export_audacity, requires_full=True)

    section("LoRA (TTS)")
    run_test("lora_test_model", test_lora_test_model, requires_full=True)
    run_test("lora_generate_dataset", test_lora_generate_dataset, requires_full=True)

    section("Dataset Builder Generate (TTS)")
    run_test("dataset_builder_generate_sample", test_dataset_builder_generate_sample, requires_full=True)


# ── Cleanup ──────────────────────────────────────────────────

def main():
    global BASE_URL, FULL_MODE

    parser = argparse.ArgumentParser(description="Threadspeak API test suite")
    parser.add_argument("--full", action="store_true",
                        help="Include TTS/LLM-dependent tests")
    args = parser.parse_args()

    _assert_no_external_server_target()
    FULL_MODE = args.full

    _start_isolated_test_server()

    print(f"Threadspeak API Tests")
    print(f"Server: {BASE_URL}")
    print(f"Mode:   {'FULL (includes TTS/LLM tests)' if FULL_MODE else 'QUICK (no TTS/LLM)'}")

    cleanup()

    try:
        run_all_tests()
    finally:
        cleanup()
        _stop_isolated_test_server()

    # Summary
    total = results["passed"] + results["failed"] + results["skipped"]
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {results['passed']} passed, {results['failed']} failed, "
          f"{results['skipped']} skipped  (total: {total})")
    print(f"{'=' * 60}")

    if failures:
        print(f"\nFailed tests:")
        for name, err in failures:
            # Truncate long error messages
            short = err.split("\n")[0][:200]
            print(f"  - {name}: {short}")

    sys.exit(1 if results["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
