# ThakAI – Arabic Legal Document Pipeline: Strategy & Architecture

> This document defines the technical strategy for ingesting, extracting, and
> querying Arabic legal documents. It covers the OCR approach, pipeline
> orchestration, model selection, fine-tuning plan, and deployment recommendations.

---

## 1. Core Architectural Decisions

### 1.1 OCR: VLM-First, Not Classical OCR

Arabic legal PDFs fall into two categories: digital-born PDFs with selectable
Unicode text, and scanned/image-based PDFs where text must be extracted visually.

For the second category, the correct tool is a **Vision-Language Model (VLM)**,
not a classical OCR engine like Amazon Textract. The reasons are specific to Arabic:

- Arabic is cursive, right-to-left, and uses diacritics that classical engines
  were not designed to handle
- Legal documents mix Arabic and English, contain stamps, seals, handwritten
  signatures, and structured tables — all of which VLMs handle in a single pass
- A VLM reads the page image exactly as a human would, producing structured JSON
  output that includes text, layout, metadata, and visual elements simultaneously

Classical OCR is retained only as a last-resort fallback.

**Extraction decision tree:**

```
PDF received from S3
    │
    ▼
PyMuPDF: attempt Unicode text extraction
    │
    ├── char yield > threshold ──────────────────▶ Use extracted text directly
    │   (digital-born PDF)
    │
    └── char yield below threshold (scanned PDF)
            │
            ▼
        Render pages as images
        Preprocess: grayscale → 600px max width → 1.5× contrast enhancement
            │
            ▼
        VLM Extraction (one call per page, full structured JSON output)
```

**Image preprocessing standards:**
- Convert to grayscale (reduces token cost, improves consistency)
- Resize to 600px maximum width maintaining aspect ratio
- Apply 1.5× contrast enhancement (improves legibility on faded printed pages)

These settings are fixed defaults across all VLM backends.

---

### 1.2 Extraction: Single Structured Prompt, Not Multi-Agent

The most important design decision in the extraction layer is this: **one VLM call
per page extracts everything** — OCR text, document classification, metadata, dates,
legal articles, tables, charts, stamps, seals, signatures, and quality confidence.

This is the **Master Extraction Prompt** approach. It is preferred over a
multi-agent pipeline for the extraction layer for clear reasons:

| | Single Structured Prompt | Multi-Agent Crew |
|---|---|---|
| API calls per page | 1 | 3–5 |
| Cost | Baseline | 3–5× higher |
| Error propagation | None | Each agent inherits previous errors |
| Determinism | Yes — one input, one output | No — agents can disagree or retry |
| Testability | One unit test per page | Complex trace required |
| Suitable for | Structured ETL with known schema | Open-ended reasoning tasks |

A full autonomous multi-agent Crew is reserved for the **downstream QA layer**,
where legal reasoning, cross-document synthesis, and open-ended question answering
genuinely benefit from specialist agent collaboration.

---

### 1.3 Pipeline Orchestration: CrewAI Flows

The extraction pipeline itself uses **CrewAI Flows** — deterministic, event-driven
orchestration where each stage fires once, receives state from the previous stage,
makes one LLM call, and passes results forward.

This is distinct from an autonomous Crew. There are no agent loops, no delegation,
no retries. The benefit is clean per-stage model assignment (different LLM per task)
and LiteLLM's unified interface supporting Bedrock, vLLM, and Ollama without
rewriting logic.

```
CrewAI Flow
  @start      → Pre-processing (VLM OCR if needed)
  @listen     → Stage 1: Page splitting and assembly
  @listen     → Stage 2: Document profiling        [Falcon-Arabic 7B]
  @listen     → Stage 3: Structural extraction     [Falcon-Arabic 7B]
  @listen     → Stage 4: Metadata consolidation    [Falcon-Arabic 7B / Jais 30B]
  └── writes to PostgreSQL → triggers embeddings

CrewAI Crew (multi-agent, downstream only)
  Agent 1: Legal Article Retriever   [Falcon-Arabic 7B]
  Agent 2: Legal Analyst             [Jais 30B]
  Agent 3: Arabic Responder          [Falcon-Arabic 7B]
```

---

## 2. The Master Extraction Prompt

A single comprehensive prompt drives all VLM extraction. It is designed specifically
for Arabic legal documents and extracts the following in one pass:

- **Full text** — complete page content in original Arabic script, with line breaks
- **Document classification** — type, category, primary language
- **Source metadata** — issuing authority, document number, department
- **Dates** — primary header date and all additional dates, with Hijri/Gregorian
  calendar identification
- **Legal articles** — article number, title, and full content
- **Tables** — headers and row data in structured arrays
- **Charts** — data points extracted as labelled arrays, not text summaries
- **Structural elements** — header, footer, letterhead content
- **Official marks** — seals (circular emblems) and stamps (ink impressions)
  correctly differentiated, with position and text
- **Signatures** — signatory names and titles in original Arabic script,
  signature type (handwritten / digital / stamp)
- **Quality assessment** — overall confidence level and review flags

**Critical extraction rules enforced by the prompt:**

1. All Arabic text stays in Arabic — names and titles are never romanised or translated
2. Document numbers are extracted exactly as shown: `"13/ت/8795"` not `"13-8795"`
3. Primary date (header) is separated from additional dates referenced in the body
4. Chart data is always a structured array — never collapsed into a text summary
5. Output is pure JSON starting with `{` and ending with `}` — no markdown wrapper

---

## 3. Fine-Tuning Strategy

### 3.1 Why Fine-Tune

General-purpose VLMs perform well on Arabic documents out of the box. However,
ThakAI's corpus has specific characteristics that a fine-tuned model handles
significantly better:

- UAE-specific issuing authorities and their Arabic name variants
- UAE federal law numbering format (e.g., `مرسوم اتحادي رقم (X) لسنة XXXX`)
- Hijri date formats as used in UAE government documents
- Specific stamp and seal layouts from UAE ministries
- Mixed Arabic-English headers common in UAE regulatory documents

A fine-tuned **Gemma 3 4B** model trained on ThakAI's own document corpus will
outperform all general-purpose models on these patterns because it has seen them
during training.

### 3.2 Knowledge Distillation Approach

Fine-tuning data is generated via **knowledge distillation**: a large, capable
cloud model (the teacher) labels the training data, and a smaller efficient model
(the student) learns from those labels.

```
ThakAI Arabic PDFs
        │
        ▼
PDF → image pages (preprocessed: grayscale, 600px, 1.5× contrast)
        │
        ▼
Teacher model: Gemini 2.5 Flash
Runs Master Extraction Prompt on each page
Outputs structured JSON per page
        │
        ▼
Labelled dataset: (page_image, extraction_json) pairs → JSONL file
        │
        ▼
Split into two SFT tasks per record:
  Task 1 → content + structural_elements   (OCR and layout)
  Task 2 → classification, source, dates, signatures,
           official_marks, routing, condition_notes,
           confidence_quality                (document metadata)
        │
        ▼
Fine-tune student model: Gemma 3 4B (LoRA)
        │
        ▼
Deployed fine-tuned model replaces teacher for production inference
```

**Cost estimate for ThakAI corpus:**

| Corpus size | Labelling cost (Gemini Flash) | Training (EC2 g5.2xlarge spot) |
|---|---|---|
| 2,000 pages | ~$18 | ~$8 |
| 5,000 pages | ~$45 | ~$15 |
| 10,000 pages | ~$90 | ~$25 |

This is a one-time cost. The fine-tuned model then runs at ~$1/hr on EC2 g5.xlarge
with no per-page API cost.

### 3.3 Fine-Tuning Configuration

**Framework:** LlamaFactory (supports Gemma 3 vision models natively)  
**Method:** LoRA (Low-Rank Adaptation) — trains adapter weights only, base model unchanged

```yaml
model:         google/gemma-3-4b-it
method:        LoRA
lora_rank:     96
lora_target:   all
cutoff_len:    12,000 tokens
epochs:        20
learning_rate: 1e-4
scheduler:     cosine with 10% warmup
batch:         1 per device × 8 gradient accumulation steps (effective batch = 8)
precision:     bf16
eval_steps:    every 50 steps
```

**Hardware requirements:**
- Training: single A100 40GB or 2× A10G (AWS g5.2xlarge or g5.12xlarge)
- Inference after training: single g5.xlarge (~$1/hr) — model fits in 24GB VRAM

### 3.4 Deployment of Fine-Tuned Model

After training, LoRA adapter weights are merged into the base model and deployed
on AWS via vLLM, which exposes an OpenAI-compatible API. The pipeline's OCR backend
configuration is switched from the base model to the fine-tuned endpoint — no other
code changes required.

**Break-even vs. API inference:**
At $0.009/page (Gemini Flash) vs. $1/hr (EC2 g5.xlarge serving ~200 pages/hr),
the fine-tuned model breaks even at roughly 110 pages/day and becomes progressively
cheaper at higher volumes.

---

## 4. Model Selection

### 4.1 VLM / OCR Models

| Model | Type | Arabic Quality | Cost | Use case |
|---|---|---|---|---|
| **Gemma 3 4B (fine-tuned)** | Open, 4B | Domain-best | ~$1/hr EC2 | Primary production model after fine-tuning |
| **Gemma 3 27B (base)** | Open, 27B | Excellent | ~$5.70/hr SageMaker | Production before fine-tune is ready; strong fallback |
| **QARI-OCR v0.2** | Open, 2B | CER 0.061 — best open | ~$1/hr EC2 | Open-source Arabic OCR specialist (Qwen2-VL backbone) |
| **Mistral OCR API** | API | Good | 1,000 pages/$ | Managed option; no GPU; table-aware |
| **Gemini 2.5 Flash** | API | Excellent | ~$0.009/page | Knowledge distillation teacher; API fallback |
| **Claude 3.5 Sonnet** | API (Bedrock) | Excellent | ~$0.005/page | AWS-native managed fallback |

### 4.2 Text LLMs (Stages 2–4, downstream QA)

| Model | Origin | Strengths | Context |
|---|---|---|---|
| **Falcon-Arabic 7B** | UAE — TII, Abu Dhabi | Outperforms 28B+ on OALL v2 Arabic benchmark; UAE legal domain native | 32k |
| **Jais 30B** | UAE — MBZUAI / G42 | Purpose-built for UAE legal and cultural context; deep Arabic reasoning | 8k |
| **Gemma 3 27B** | Google | Strong multilingual; 128k context fits entire documents in one pass | 128k |
| **Qwen3 30B** | Alibaba | Top OALL v2 ranking; strong structured JSON output; 128k context | 128k |
| **Claude 3.5 Sonnet** | Anthropic (Bedrock) | Managed; no GPU; AWS IAM-integrated; best if operational simplicity > cost | 200k |

**Recommended assignment:**

| Stage | Model | Reason |
|---|---|---|
| Document profiling (Stage 2) | Gemma 3 27B or Qwen3 30B | 128k context; strong structured output |
| Structure + metadata (Stages 3–4) | Falcon-Arabic 7B | UAE-native; fast; 7B parameters sufficient for structured tasks |
| Legal QA (deep reasoning) | Jais 30B | UAE legal domain; purpose-built for this context |
| Legal QA (high throughput) | Falcon-Arabic 7B | Cheapest per-token with strong Arabic quality |

### 4.3 Embedding Models

| Model | Cross-lingual | Access | Notes |
|---|---|---|---|
| **Cohere Embed Multilingual v3** | Yes (Arabic↔English) | Amazon Bedrock | Best for cross-lingual retrieval; managed |
| **GATE** | Arabic-native | HuggingFace (self-host) | Highest Arabic semantic accuracy |
| **Qwen3 Embedding** | Yes | HuggingFace / API | Strong multilingual; open-source |

---

## 5. Full Pipeline Architecture

```
PDF from S3
    │
    ▼
PyMuPDF: Unicode text extraction attempt
    │
    ├── [Digital PDF — selectable text]
    │       └── text → Arabic normalisation → Stage 1
    │
    └── [Scanned PDF — low text yield]
            │
            ▼
        Render pages as images
        Preprocess: grayscale, 600px, 1.5× contrast
            │
            ▼
        ┌────────────────────────────────────────────────────┐
        │  VLM EXTRACTION  (one call per page)               │
        │                                                    │
        │  Production                                        │
        │  └─ Gemma 3 4B (fine-tuned on ThakAI corpus)      │
        │                                                    │
        │  Fallback / before fine-tune                       │
        │  └─ Gemma 3 27B  |  QARI-OCR  |  Mistral OCR API  │
        │     Claude 3.5 Sonnet (Bedrock)                    │
        └────────────────────────────────────────────────────┘
            │
            ▼
        Structured JSON per page → assembled document
            │
            ▼
    ─── CrewAI Flow (deterministic) ─────────────────────────
    Stage 1: Page splitting and assembly
    Stage 2: Document profiling          [Gemma 3 27B]
    Stage 3: Structural node extraction  [Falcon-Arabic 7B]
    Stage 4: Metadata consolidation      [Falcon-Arabic 7B]
            │
            ▼
    PostgreSQL (RDS)
    documents_raw | document_pages | document_nodes | document_metadata
            │
            ▼
    Embeddings: Cohere Multilingual v3 (Bedrock)
            │
            ▼
    Amazon OpenSearch Serverless (hybrid BM25 + vector)
            │
            ▼
    ─── CrewAI Crew (multi-agent, QA layer) ──────────────────
    Agent 1: Legal Retriever    [Falcon-Arabic 7B]
    Agent 2: Legal Analyst      [Jais 30B]
    Agent 3: Arabic Responder   [Falcon-Arabic 7B]
            │
            ▼
    Final answer with article citations (Arabic or English)
```

---

## 6. AWS Deployment

All open-source models are deployable within AWS without data leaving the environment.

| Model | Deployment method | Instance | Estimated cost |
|---|---|---|---|
| Gemma 3 4B (fine-tuned) | EC2 + vLLM | g5.xlarge | ~$1.00/hr |
| QARI-OCR v0.2 | EC2 + vLLM | g5.xlarge | ~$1.00/hr |
| Falcon-Arabic 7B | SageMaker TGI endpoint | g5.2xlarge | ~$1.20/hr |
| Gemma 3 27B | SageMaker JumpStart | g5.12xlarge | ~$5.70/hr |
| Jais 30B | EC2 + vLLM | g5.12xlarge | ~$5.70/hr |
| Qwen3 30B | SageMaker + vLLM | g6.12xlarge | ~$5.00/hr |

All models listed carry Apache 2.0 or equivalent commercially-usable licenses.

**Batch processing recommendation:** run EC2 Spot instances (70–80% cost reduction)
for bulk ingestion of archived documents, with per-document checkpointing so that
spot interruptions resume from the last completed document.

---

## 7. Recommended Stacks

### Production Stack (balanced cost and quality)

| Layer | Tool |
|---|---|
| OCR — production | Gemma 3 4B fine-tuned, EC2 g5.xlarge + vLLM |
| OCR — pre-fine-tune | QARI-OCR v0.2 or Gemma 3 27B base |
| Pipeline orchestration | CrewAI Flows |
| Profiling (Stage 2) | Gemma 3 27B, SageMaker JumpStart |
| Structure + metadata (Stages 3–4) | Falcon-Arabic 7B, SageMaker |
| Legal QA | Jais 30B (deep) + Falcon-Arabic 7B (fast) |
| Embeddings | Cohere Embed Multilingual v3, Amazon Bedrock |
| Vector store | Amazon OpenSearch Serverless |

### Managed Stack (no GPU infrastructure)

| Layer | Tool |
|---|---|
| OCR | Mistral OCR API |
| All text stages | Claude 3.5 Sonnet, Amazon Bedrock |
| Embeddings | Cohere Embed Multilingual v3, Amazon Bedrock |
| Vector store | Amazon OpenSearch Serverless |

---

## 8. Implementation Roadmap

| Phase | Milestone | Primary model |
|---|---|---|
| 1 | Digital PDFs ingested; metadata extracted | Falcon-Arabic 7B (SageMaker) |
| 2 | Scanned PDF support via VLM extraction | Gemma 3 27B or QARI-OCR v0.2 |
| 3 | CrewAI Flow orchestration deployed | — |
| 4 | Fine-tuning dataset generated via knowledge distillation | Gemini 2.5 Flash (teacher) |
| 5 | Gemma 3 4B LoRA fine-tuned on ThakAI corpus | LlamaFactory on EC2 |
| 6 | Fine-tuned model deployed; replaces base model in OCR backend | EC2 g5.xlarge + vLLM |
| 7 | Embeddings + OpenSearch + RAG QA API live | Cohere + Jais 30B QA Crew |

---

## 9. Decision Summary

| Decision | Choice | Rationale |
|---|---|---|
| OCR engine | VLM-first | Arabic cursive script, diacritics, and mixed layouts require vision understanding |
| Extraction architecture | Single structured prompt per page | One call extracts all fields; deterministic; cheapest; fully testable |
| Pipeline orchestration | CrewAI Flows | Deterministic stage routing with per-stage model assignment; no agent overhead |
| Multi-agent use | QA layer only | Autonomous collaboration adds value for open-ended legal reasoning, not ETL |
| Fine-tuning model | Gemma 3 4B | Natively multimodal; Arabic-capable; runs on single g5.xlarge; commercially licensed |
| Fine-tuning method | Knowledge distillation + LoRA | ~$45 labelling cost for 5k pages; no human annotation required |
| UAE legal domain model | Falcon-Arabic 7B / Jais 30B | UAE-native training data; Falcon outperforms models 4× its size on OALL v2 |
| Embeddings | Cohere Multilingual v3 (Bedrock) | Best Arabic↔English cross-lingual retrieval; fully managed |
