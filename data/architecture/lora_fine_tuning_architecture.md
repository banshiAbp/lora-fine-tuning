# LoRA Fine-Tuning Architecture

## Offline Training Path

```text
Databricks Dolly-15K
        |
        v
Schema and null validation
        |
        v
Deterministic shuffle (seed 42)
        |
        +----> 1,000 train examples
        |
        `----> 100 held-out examples
                    |
                    v
Conversational messages: user instruction/context -> assistant response
                    |
                    v
TinyLlama chat template + 512-token truncation
                    |
                    v
Frozen TinyLlama-1.1B weights
        + LoRA A/B matrices on q_proj and v_proj
                    |
                    v
SFTTrainer: cosine schedule, 3% warmup, effective batch 16, one epoch
                    |
                    v
Adapter weights + config + tokenizer + reproducibility manifest
```

Only the low-rank matrices receive gradient updates. The notebook asserts that the trainable share is below 1% and audits every trainable tensor name.

## Evaluation Path

```text
Required prompts -----------------> frozen base responses
       |                                      |
       |                                      v
       +--------------------------> deterministic comparison table
       |                                      ^
       v                                      |
reloaded base + saved LoRA adapter -> tuned responses

Held-out Dolly examples -> pre-SFT loss/perplexity
                       -> post-SFT loss/perplexity
                       -> metric delta and interpretation
```

## Deployment Path

```text
TinyLlama base model (shared once)
        +
small client-specific LoRA adapter
        |
        v
PEFT model for client-specific inference
```

This separation lets Orion AI Labs maintain one base model and switch lightweight adapters for different client styles or terminology.
