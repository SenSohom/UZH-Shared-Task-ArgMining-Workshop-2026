# ArgMining 2026 — Generative Claim-Evidence Pipeline

**System B** submission for the [UZH ArgMining 2026 Shared Task](https://github.com/ZurichNLP/ArgMining-2026-UZH-Shared-Task) — recovering argumentative structure from UN/UNESCO resolutions.

The pipeline adapts a **claim-evidence-strategy framework** to resolution analysis, classifying paragraphs and predicting argumentative relations with explicit reasoning traces (`think` fields). Generation runs locally on **Qwen3-8B-GGUF**; evaluation uses **Claude Sonnet 4.6** as an LLM-as-a-Judge.

| Split | LLM-Judge Score |
|-------|----------------|
| Train (2,694 docs) | 81 / 100 |
| Test  (89 docs)    | 77 / 100 |

---

## How It Works

The pipeline runs in two sequential steps per resolution document:

**Step 1 — Paragraph Classification**
- Classifies each paragraph as `preambular` (contextual/rationale) or `operative` (directive)
- Assigns education-dimension tags from a controlled 126-code vocabulary
- Uses a sliding window of ±21 paragraphs centred on the target for context
- Produces a 4-sentence structured `think` field grounded in verbatim quotes

**Step 2 — Relation Prediction**
- Retrieves top-15 semantically related candidate paragraphs via `sentence-transformers`
- Predicts argumentative relations: `supporting`, `complemental`, `contradictive`, `modifying`
- Each relation maps to a named reasoning strategy (Causal / Corroboration / Contrastive / Triangulation)
- Produces a 5-sentence `think` field with quoted evidence and an explicitly rejected alternative

Non-argumentative paragraphs (no tags from Step 1) are filtered out before Step 2.

---

## Requirements

```
python >= 3.9
anthropic
sentence-transformers
scikit-learn
numpy
json-repair
```

Install dependencies:
```bash
pip install anthropic sentence-transformers scikit-learn numpy json-repair
```

### Local inference (Qwen3-8B-GGUF)

Download [Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) and serve it via [llama.cpp](https://github.com/ggerganov/llama.cpp) with an OpenAI-compatible endpoint:

```bash
./llama-server -m qwen3-8b-q4_k_m.gguf --port 8080 -ngl 99
```

Update `STEP1_MODEL` / `STEP2_MODEL` in `config.py` and point the `anthropic.Anthropic` client in `claims_pipeline.py` to `base_url="http://localhost:8080/v1"`.

---

## Setup

### 1. Dataset

Download the shared task data:
```bash
git clone https://huggingface.co/datasets/ZurichNLP/ArgMining-2026-UZH-Shared-Task
```

Set the path in `config.py` or via environment variable:
```bash
export ARGMINING_DATA="/path/to/ArgMining-2026-UZH-Shared-Task"
```

### 2. API Key (judge only)

The Anthropic API key is only required for evaluation. Generation runs locally.
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Usage

### Run on test data (submission)
```bash
python claims_pipeline.py --test
```

### Run on train data
```bash
python claims_pipeline.py
```

### Process a limited number of documents (development)
```bash
python claims_pipeline.py --test --limit 5
```

Already-processed files are skipped automatically — safe to resume interrupted runs.

### Custom input folder
```bash
python claims_pipeline.py --input /path/to/json/folder
```

---

## Evaluation

### LLM-as-a-Judge (no gold labels needed)
```bash
python evaluate.py --pred output/my_doc_predictions.json --judge-only

# Entire output directory
python evaluate.py --pred output/ --judge-only
```

### Full evaluation with gold labels (F1 + Judge)
```bash
python evaluate.py --pred output/ --gold /path/to/gold/
```

---

## Output Schema

Each prediction file follows the official shared task schema:

```json
{
  "TEXT_ID": "ICPE-03-1934_RES1-FR_res_1",
  "METADATA": {
    "structure": {
      "preambular_para": [1, 2, 3],
      "operative_para": [4, 5, 6, 7],
      "think": "Document-level reasoning..."
    }
  },
  "body": {
    "paragraphs": [
      {
        "para_number": 1,
        "para": "Original French text...",
        "para_en": "English translation...",
        "type": "preambular",
        "tags": ["ACT_IO", "LAW_INTER"],
        "matched_pars": {"4": ["supporting"], "6": ["complemental"]},
        "think": "Step 1 reasoning...\n\n[→ para 4] Step 2 reasoning..."
      }
    ]
  }
}
```

---

## Reasoning Strategy → Relation Type Mapping

| Reasoning Strategy | Relation Type  | Description |
|--------------------|---------------|-------------|
| Causal             | `supporting`   | Premise → directive justification |
| Corroboration      | `complemental` | Same claim, different evidence/actors |
| Contrastive        | `contradictive`| Opposing or limiting positions |
| Triangulation      | `modifying`    | Conditions, exceptions, or scope refinement |

---

## Repository Structure

```
├── claims_pipeline.py   # Main pipeline (classification + relation prediction)
├── config.py            # Models, paths, prompts, tag vocabulary loader
├── evaluate.py          # LLM-as-a-Judge + F1 evaluation (System B)
├── evaluate_sysA.py     # Evaluation for System A predictions
└── paper/               # ACL-format system paper (LaTeX)
```

---

## Citation

If you use this code, please cite the shared task:

```bibtex
@inproceedings{uzh2026argmining,
  title     = {UZH ArgMining 2026 Shared Task},
  booktitle = {Proceedings of the ArgMining Workshop 2026},
  year      = {2026}
}
```

