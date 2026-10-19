# Local, Colab, and Docker Execution Guide

## Recommended: Google Colab T4

1. Upload or open `notebooks/LoRA_Assignment.ipynb`.
2. Choose **Runtime -> Change runtime type -> T4 GPU**.
3. If any earlier command upgraded `torch`, `torchvision`, or downgraded `huggingface-hub`, choose **Runtime -> Disconnect and delete runtime**. Reconnect to a T4; an ordinary restart does not restore the original packages.
4. Run all cells in order. Do not skip the baseline cells: their responses and loss are used by later comparisons.
5. Download the executed notebook and, if needed, `sft-lora-tinyllama/final-adapter/`.

If a CUDA out-of-memory error occurs, restart the runtime and run all again. The assignment's batch size is fixed at 4; do not silently change it in the submitted run.

## Local Windows Setup

```powershell
cd C:\python\ML-POC\lora-fine-tuning
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-local.txt
python -m jupyter lab notebooks\LoRA_Assignment.ipynb
```

Check GPU visibility before training:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

## Docker with NVIDIA GPU

Docker execution requires Docker Desktop, WSL2, a compatible NVIDIA driver, and the NVIDIA Container Toolkit.

```powershell
docker build -t tinyllama-lora-gpu .
docker run --rm -it --gpus all `
  -p 8888:8888 `
  -v "${PWD}:/workspace" `
  -v tinyllama-hf-cache:/root/.cache/huggingface `
  tinyllama-lora-gpu
```

Open the Jupyter URL printed by the container and run `/workspace/notebooks/LoRA_Assignment.ipynb`.

## Output Locations

```text
sft-lora-tinyllama/
|-- final-adapter/
|   |-- adapter_config.json
|   |-- adapter_model.safetensors
|   `-- tokenizer files
|-- run_manifest.json
`-- response_comparison.json
```

These outputs are ignored by Git. Copy them into `artifacts/local_checkpoint/` only for local archival; do not commit generated model files.

## Common Errors

### CUDA is unavailable

In Colab, select a T4 runtime and restart. Locally, confirm the NVIDIA driver and CUDA-enabled PyTorch build are installed.

### `torchvision::nms does not exist` or torchvision requires a different torch version

The Colab CUDA pair was changed by pip. Choose **Runtime -> Disconnect and delete runtime**, reconnect to a T4, and run the updated notebook from the top. Do not install or upgrade `torch` or `torchvision` in Colab.

### PEFT reports an incompatible `torchao` version during `get_peft_model`

TorchAO is an optional quantization package and is not used by this FP16 LoRA assignment. Some Colab images include an older TorchAO release that current PEFT detects and rejects. The updated setup cell removes TorchAO before importing PEFT. Delete the current runtime, reconnect, and run the notebook from the first cell so the uninstall happens before PEFT is imported.

### `huggingface-hub` conflicts with Diffusers or Gradio

This was caused by constraining Transformers to an older major version while Colab had newer Hugging Face applications installed. The updated notebook installs Transformers, TRL, PEFT, Datasets, and Accelerate together without downgrading the Hub package. Start with a deleted/fresh runtime to remove the old downgrade.

### `SFTConfig` or `SFTTrainer` rejects an argument

The notebook now inspects the installed `SFTConfig` signature. It uses `warmup_ratio=0.03` where supported and the equivalent calculated `warmup_steps` otherwise; it also handles `eval_strategy`/`evaluation_strategy` and `max_length`/`max_seq_length` naming changes. Delete the runtime only when packages were changed after imports. For a Task 6 configuration error, use the updated cell and rerun it directly.

### Out of memory

Restart the runtime to clear orphaned allocations. Confirm no earlier duplicate model objects remain. The notebook deletes the inspection model before training and clears CUDA memory.

### BF16 GEMM requires compute capability 8.0 or higher

T4 GPUs have compute capability 7.5 and must use FP16. Some recent PyTorch builds can report BF16 availability even when a training kernel cannot execute BF16 on that GPU. The updated Task 6 checks the actual CUDA compute capability: it enables BF16 only on 8.0+ hardware and otherwise sets `bf16=False`, `fp16=True`.

### Adapter reload mismatch

Confirm greedy decoding remains enabled and that the reload cell points to the same `ADAPTER_DIR` created by the save cell.

### Hugging Face download fails

Retry once, verify internet access, and set `HF_TOKEN` securely only if authentication is required. Never paste a token into a committed cell.
