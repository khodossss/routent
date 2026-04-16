# Training Configs

Each config defines a complete training run: model pool, dataset, evaluation mode, and reward weights.

Run any config with:
```bash
python routent/scripts/train.py --config configs/train/<config>.json
```

---

## gsm8k_gpt-5-nano.json

**Task:** Grade-school math (GSM8K). Route between the same model at different `reasoning_effort` levels (minimal / low / medium).

**Eval:** `numeric` — extracts the final number from the model's response and compares to the ground truth. Auto-generated prompt suffix asks for only the numeric answer.

**Expected:** Router learns that `minimal` fails on multi-step problems, `low` is the sweet spot, `medium` is overkill for easy ones. Quality is binary (correct number or not).

---

## gsm8k_mixed.json

**Task:** Same GSM8K math, but with a heterogeneous model pool — cloud APIs (OpenAI, Google) mixed with local HuggingFace models.

**Eval:** `numeric` (binary).

**Expected:** Router learns that local models are free but less accurate, cloud models cost money but solve harder problems. Tests cost/quality tradeoff across providers.

---

## gsm8k_gemini.json

**Task:** GSM8K with Google Gemini models.

**Eval:** `numeric` (binary).

**Expected:** Baseline for comparing Google provider routing behavior.

---

## gsm8k_fuzzy.json

**Task:** GSM8K evaluated on full reasoning text (not just the final number). Custom `prompt_suffix` asks the model to show step-by-step work.

**Eval:** `fuzzy` — SequenceMatcher ratio between the full model response and the ground truth solution. **Continuous** quality [0, 1].

**Expected:** Models that produce reasoning closer to the reference solution get higher quality, even if the final number is slightly different. Tests continuous reward signal.

---

## sst2_sentiment.json

**Task:** Binary sentiment classification (SST-2: positive/negative). Routes between OpenAI API and a local DistilBERT classifier.

**Eval:** `classification` with `choices: ["positive", "negative"]`. Auto-generated prompt suffix: "Answer with ONLY one of: positive, negative". **Binary** quality.

**Expected:** Router learns that DistilBERT is free and fast for easy sentiment, but OpenAI handles ambiguous cases better. Tests API vs local model routing.

---

## sst2_confidence.json

**Task:** Same SST-2, but evaluated on model **confidence** instead of binary correctness.

**Eval:** `classification` + `output_format: "confidence_json"`. The model responds with `{"positive": 0.92, "negative": 0.08}`. Quality = probability assigned to the correct class. **Continuous** [0, 1].

**Expected:** A model that answers "positive" with 51% confidence gets quality=0.51, not 1.0. Router learns to send uncertain queries to a stronger model.

---

## mmlu_classification.json

**Task:** MMLU multiple-choice (A/B/C/D) across many knowledge domains. Routes between OpenAI and a local zero-shot classifier (BART-MNLI).

**Eval:** `classification` with `choices: ["A", "B", "C", "D"]`. **Binary** quality.

**Expected:** Router discovers which subjects the zero-shot model handles (common sense) vs which need GPT (specialized knowledge like law, medicine).

---

## goemotions_multilabel.json

**Task:** Multi-label emotion detection (GoEmotions). Each text can have multiple emotions (e.g., "joy,admiration").

**Eval:** `multilabel` — Jaccard similarity between predicted and ground truth label sets. **Continuous** quality: getting 2 out of 3 labels right = 0.67, not 0.

**Expected:** Router learns which model is better at detecting multiple overlapping emotions. Partial credit gives LinUCB a richer signal than binary all-or-nothing.

---

## stsb_regression.json

**Task:** Semantic textual similarity (STS-B). Given two sentences, predict similarity score 0-5.

**Eval:** `regression` with `tolerance: 1.0`. Quality = `max(0, 1 - |error| / 1.0)`. **Continuous**: predicting 3.5 when the answer is 4.0 gives quality=0.5, not 0.

**Expected:** Router learns which model estimates similarity more precisely. Tests continuous regression reward.

---

## mt_bench_semantic.json

**Task:** Open-ended questions (MT-Bench) evaluated by embedding similarity to a reference answer.

**Eval:** `semantic` — cosine similarity between sentence embeddings of the model response and reference. **Continuous** [0, 1].

**Expected:** Router learns which model produces responses semantically closest to the reference. No LLM judge needed — fast, deterministic evaluation.

---

## mt_bench_llm_judge.json

**Task:** Open-ended questions (MT-Bench) evaluated by an LLM judge (GPT-5-nano).

**Eval:** `llm_judge` — binary YES/NO correctness from the judge. **Binary** quality.

**Expected:** Baseline for LLM-judge evaluation. The judge decides if the answer is acceptable or not.

---

## mt_bench_weighted_criteria.json

**Task:** Same MT-Bench, but the LLM judge scores on **multiple weighted criteria** instead of binary YES/NO.

**Eval:** `llm_judge` with weighted criteria dict:
```json
{"correctness": 0.5, "completeness": 0.3, "conciseness": 0.2}
```
Each criterion scored 1-10, normalized to [0, 1], weighted and averaged. **Continuous** quality.

**Expected:** Router learns multi-dimensional quality: model A might be correct but verbose (high correctness, low conciseness), model B might be concise but incomplete. The weighted combination guides routing.
