# Model Recommendations

Threadspeak uses LLMs in multiple stages (for example dialogue attribution, temperament extraction, and voice suggestion tasks).

## General guidance

- Prefer reliable instruction-following models.
- Avoid highly verbose reasoning output in stages that expect strict tool/format behavior.
- If a model emits unwanted control text, use banned-token controls in Setup.

## Common choices

- Qwen family models
- Gemma family models
- Llama 3.x instruct variants
- Mistral/Mixtral instruct variants

Model quality and behavior can vary by provider/runtime. Validate on a representative sample chapter before long runs.
