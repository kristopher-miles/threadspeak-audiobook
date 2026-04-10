# First Launch

If this is your first time running Threadspeak, use this checklist.

## 1. Start an LLM server first

Threadspeak connects to an OpenAI-compatible API. It does not bundle an LLM.

| Server | Default URL | Notes |
|--------|-------------|-------|
| [LM Studio](https://lmstudio.ai/) | `http://localhost:1234/v1` | Load a model, then start server |
| [Ollama](https://ollama.ai/) | `http://localhost:11434/v1` | Example: `ollama run qwen3` |
| [OpenAI API](https://platform.openai.com/) | `https://api.openai.com/v1` | Requires API key |

If the LLM is not reachable, ingestion/attribution stages fail.

## 2. Expect first-time model downloads

Qwen3-TTS model variants download on first use (about 3.5 GB each). The UI may appear idle while this happens.

## 3. Expect first-batch warmup

First batch in a session is slower due to one-time warmup work:
- AMD MIOpen autotuning
- Optional codec compilation warmup

## 4. Tune for available VRAM

| Available VRAM | Typical operation |
|---------------|-------------------|
| 8 GB | Small batches, lower parallelism |
| 16 GB | Comfortable mixed workloads |
| 24 GB+ | Higher-throughput batching |

If you hit OOM, reduce `Parallel Workers` and/or `Max Chars`.

## 5. Use terminal logs for root-cause detail

Pinokio terminal output is the best source for model-load, download, and generation errors.

For broader troubleshooting, see [Troubleshooting](Troubleshooting).
