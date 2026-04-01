# ArgMining 2026 — Generative Claim-Evidence Pipeline

**System B** submission for the [UZH ArgMining 2026 Shared Task](https://github.com/ZurichNLP/ArgMining-2026-UZH-Shared-Task) — recovering argumentative structure from UN/UNESCO resolutions.

The pipeline adapts a **claim-evidence-strategy framework** to resolution analysis, classifying paragraphs and predicting argumentative relations with explicit reasoning traces (`think` fields). Generation runs fully locally using **Qwen3-8B-GGUF (Q8_0)** via `llama-cpp-python`; no cloud API is needed to run the pipeline.

| Split | LLM-Judge Score |
|-------|----------------|
| Train (2,694 docs) | 81 / 100 |
| Test  (89 docs)    | 77 / 100 |

*Judge: Claude Sonnet 4.6 — used only for evaluation, not generation.*

---

## Files

| File | Purpose |
|------|---------|
| `config_qwen.py` | Paths, model settings, prompts, tag vocabulary loader |
| `claims_pipeline_qwen.py` | Main pipeline — classification + relation prediction |

---

## How It Works

**Step 1 — Paragraph Classification**
- Classifies each paragraph as `preambular` (contextual/rationale) or `operative` (directive)
- Assigns education-dimension tags from a 126-code controlled vocabulary
- Uses a sliding window of ±21 paragraphs centred on the target for context
- Produces a 4-sentence structured `think` field grounded in verbatim quotes

**Step 2 — Relation Prediction**
- Retrieves top-15 semantically related candidate paragraphs via `sentence-transformers`
- Predicts argumentative relations: `supporting`, `complemental`, `contradictive`, `modifying`
- Each relation maps to a reasoning strategy: Causal / Corroboration / Contrastive / Triangulation
- Produces a 5-sentence `think` field with quoted evidence and an explicitly rejected alternative

Non-argumentative paragraphs (no tags assigned in Step 1) are filtered out before Step 2.

Qwen3's internal chain-of-thought is suppressed via the `/no_think` prefix and `<think>` tag stripping, so outputs are clean JSON only.

---

## Requirements

### Hardware
- NVIDIA GPU with **≥ 16 GB VRAM** recommended (RTX 4080 tested)
- Q8_0 quantisation fits entirely in 16 GB with `n_gpu_layers=-1`

### Python packages

```bash
# llama-cpp-python with CUDA support (required)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# remaining dependencies
pip install huggingface_hub sentence-transformers scikit-learn numpy json-repair
```

---

## Setup

### 1. Dataset

Clone the shared task data into the **same directory as the scripts**:

```bash
git clone https://huggingface.co/datasets/ZurichNLP/ArgMining-2026-UZH-Shared-Task
```

The scripts expect:
```
ArgMining-2026-UZH-Shared-Task/
├── train-data/
├── test-data/
└── education_dimensions_updated.csv
```

### 2. Output path

Open `config_qwen.py` and update `OUTPUT_BASE_PATH` to a directory on your machine:

```python
OUTPUT_BASE_PATH = '/your/path/to/output'
```

### 3. HuggingFace token (optional)

The model (`Qwen/Qwen3-8B-GGUF`) is public. A token is only needed if you have gated model access. Set it via:

```bash
export HF_TOKEN="hf_..."
```

On first run the Q8_0 GGUF file (~8.6 GB) is downloaded automatically and cached at `~/.cache/huggingface/hub/`.

---

## Usage

### Run on test data (submission)
```bash
python claims_pipeline_qwen.py --test
```

### Run on train data
```bash
python claims_pipeline_qwen.py
```

### Process a limited number of documents (development / sanity check)
```bash
python claims_pipeline_qwen.py --test --limit 5
```

### Custom input folder
```bash
python claims_pipeline_qwen.py --input /path/to/json/folder
```

Already-processed files are skipped automatically — safe to resume interrupted runs.

---

## Output Schema

Each prediction file follows the official shared task schema:

```json
{
  "TEXT_ID": "ICPE-03-1934_RES1-FR_res_1",
  "METADATA": {
    "structure": {
      "preambular_para": [1, 2, 3],
      "operative_para":  [4, 5, 6, 7],
      "think": "Document-level reasoning..."
    }
  },
  "body": {
    "paragraphs": [
      {
        "para_number": 1,
        "para":        "Original French text...",
        "para_en":     "English translation...",
        "type":        "preambular",
        "tags":        ["ACT_IO", "LAW_INTER"],
        "matched_pars": {"4": ["supporting"], "6": ["complemental"]},
        "think":       "Step 1 reasoning...\n\n[→ para 4] Step 2 reasoning..."
      }
    ]
  }
}
```

---

## Reasoning Strategy → Relation Type Mapping

| Reasoning Strategy | Relation Type   | Description |
|--------------------|-----------------|-------------|
| Causal             | `supporting`    | Premise → directive justification |
| Corroboration      | `complemental`  | Same claim, different evidence/actors |
| Contrastive        | `contradictive` | Opposing or limiting positions |
| Triangulation      | `modifying`     | Conditions, exceptions, or scope refinement |

---
```
