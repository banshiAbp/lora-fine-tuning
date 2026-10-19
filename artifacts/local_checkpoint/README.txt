LoRA Checkpoint Contents
========================

The notebook generates the following runtime artifacts:

final-adapter/adapter_model.safetensors
    Learned LoRA matrices only; the TinyLlama base weights are not duplicated.

final-adapter/adapter_config.json
    PEFT configuration, target modules, and base-model reference.

final-adapter/tokenizer files
    Tokenizer and chat-template metadata required for consistent inference.

run_manifest.json
    Hyperparameters, package versions, GPU, data counts, trainable parameter
    ratio, training/evaluation losses, and adapter size.

response_comparison.json
    Base and tuned responses for the required prompts, with transparent length
    diagnostics.

Generated artifacts are intentionally excluded from Git. This directory is a
placeholder documenting what a local checkpoint bundle should contain.
