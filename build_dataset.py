"""
ThakAI – Fine-Tuning Data Generator
Replicates the knowledge distillation pipeline from ocr_finetune_vlm.ipynb.

What this does:
  1. Converts PDF pages to preprocessed images (600px, grayscale, 1.5× contrast)
  2. Sends each image to Gemini Flash (teacher model) with the Master Extraction Prompt
  3. Saves labelled records to JSONL — the SFT training dataset
  4. Formats data for LlamaFactory LoRA fine-tuning of Gemma 3 4B

Cost reference (from notebook):
  2,058 pages × Gemini Flash → total cost $19.00
  (~$0.009/page at $0.50/1M input + $3.00/1M output tokens)

Usage:
  python finetune/build_dataset.py --pdf-dir data/pdfs/ --output data/sft/
  python finetune/build_dataset.py --format-llamafactory --sft-file data/sft/ocr-sft.jsonl
"""
import base64
import io
import json
import os
import re
import urllib.request
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# LlamaFactory YAML — paste into LlamaFactory/examples/train_lora/ocr_finetune.yaml
# ─────────────────────────────────────────────────────────────────────────────

LLAMAFACTORY_YAML = """
### model
model_name_or_path: google/gemma-3-4b-it
use_fast_tokenizer: false
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 96
lora_target: all

### dataset
dataset: ocr_finetune_train
eval_dataset: ocr_finetune_val
template: gemma3
cutoff_len: 12000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: /workspace/ocr-models-gemma-3-4b-it/
logging_steps: 25
save_steps: 50
plot_loss: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 20.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 50
report_to: wandb
""".strip()

# LlamaFactory dataset_info.json entries
DATASET_INFO = {
    "ocr_finetune_train": {
        "file_name": "/workspace/train-v1-edited.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"}
    },
    "ocr_finetune_val": {
        "file_name": "/workspace/val-v1-edited.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"}
    }
}

TASK_1_PROMPT = """
You are a professional OCR Details Extractor.
Extract: the page markdown content AND the structural_elements of the document.
Output as JSON only. No introduction or conclusion.
""".strip()

TASK_2_PROMPT = """
You are a professional OCR Details Extractor.
Extract: document_classification, source, physical_properties, official_marks,
signatures_authorization, routing_distribution, attachments_references,
condition_notes, and confidence_quality.
Output as JSON only. No introduction or conclusion.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing (from notebook cell [8])
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image_bytes(image_bytes: bytes, max_width: int = 600) -> bytes:
    """
    Preprocess a page image before sending to the VLM teacher.
    Matches notebook: grayscale → resize to 600px → 1.5× contrast.
    Returns JPEG bytes.
    """
    from PIL import Image, ImageEnhance

    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale

    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)

    img = ImageEnhance.Contrast(img).enhance(1.5)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Teacher model call (Gemini Flash via OpenRouter / Google AI API)
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini_flash(image_bytes: bytes, prompt: str) -> str | None:
    """
    Call Gemini Flash as the teacher model for knowledge distillation.
    Uses OpenRouter (as in notebook) or Google AI API directly.

    Set either:
      OPENROUTER_API_KEY + OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    or:
      GOOGLE_API_KEY
    """
    b64 = image_to_base64(image_bytes)
    data_uri = f"data:image/jpeg;base64,{b64}"

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "")

    if openrouter_key:
        # OpenRouter path (matches notebook exactly)
        payload = json.dumps({
            "model": "google/gemini-flash-1.5",
            "max_tokens": 8000,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]
            }]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type":  "application/json",
            },
            method="POST",
        )
    elif google_key:
        # Direct Google AI API
        model = "gemini-2.5-flash-preview-05-20"
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent?key={google_key}"
        )
        payload = json.dumps({
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                    {"text": prompt},
                ]
            }]
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    else:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY or GOOGLE_API_KEY to run knowledge distillation."
        )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        if openrouter_key:
            finish = data["choices"][0].get("finish_reason")
            if finish != "stop":
                return None
            return data["choices"][0]["message"]["content"]
        else:
            return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as exc:
        print(f"  Teacher call failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset building
# ─────────────────────────────────────────────────────────────────────────────

def build_sft_dataset(
    pdf_dir: str,
    output_dir: str,
    master_prompt: str,
    max_pages_per_pdf: int | None = None,
) -> str:
    """
    Run knowledge distillation: process each PDF page with the teacher model
    and save labelled records to JSONL.

    Returns path to the output JSONL file.
    """
    try:
        import fitz
        from PIL import Image
    except ImportError:
        raise ImportError("pip install PyMuPDF Pillow")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sft_file = Path(output_dir) / "ocr-sft.jsonl"

    pdfs = list(Path(pdf_dir).glob("**/*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {pdf_dir}")

    total_cost_est = 0.0
    ix = 0

    for pdf_path in pdfs:
        doc = fitz.open(str(pdf_path))
        pages = list(doc)
        if max_pages_per_pdf:
            pages = pages[:max_pages_per_pdf]

        for page in pages:
            ix += 1
            matrix = fitz.Matrix(200 / 72, 200 / 72)
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
            raw_bytes = pix.tobytes("jpeg")
            processed = preprocess_image_bytes(raw_bytes, max_width=600)

            # Save processed image
            img_dir = Path(output_dir) / "images" / pdf_path.stem
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / f"page_{page.number:04d}.jpg"
            img_path.write_bytes(processed)

            print(f"  [{ix}] {pdf_path.name} page {page.number + 1} → calling teacher...")
            llm_output = call_gemini_flash(processed, master_prompt)

            if llm_output is None:
                print(f"  [{ix}] Skipped (teacher returned None)")
                continue

            record = {
                "id": ix,
                "pdf_name": pdf_path.name,
                "image_path": str(img_path),
                "output": llm_output,
            }
            with open(sft_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Cost estimate ($0.50/1M input, $3.00/1M output — Gemini Flash)
            est_input_tokens = len(master_prompt) // 4 + 1000  # rough estimate
            est_output_tokens = len(llm_output) // 4
            total_cost_est += (est_input_tokens / 1_000_000 * 0.50 +
                               est_output_tokens / 1_000_000 * 3.00)

            if ix % 10 == 0:
                print(f"  [{ix}] Estimated cost so far: ${total_cost_est:.4f}")

        doc.close()

    print(f"\nDone. {ix} pages processed. Estimated cost: ${total_cost_est:.4f}")
    print(f"Dataset saved to: {sft_file}")
    return str(sft_file)


# ─────────────────────────────────────────────────────────────────────────────
# Format for LlamaFactory
# ─────────────────────────────────────────────────────────────────────────────

def format_for_llamafactory(
    sft_jsonl_path: str,
    output_dir: str,
    val_pdf_names: list[str] | None = None,
) -> tuple[str, str]:
    """
    Convert the raw JSONL dataset to LlamaFactory sharegpt format.
    Splits into train and val sets.
    Returns (train_json_path, val_json_path).

    Mirrors notebook cells [25]–[31].
    """
    import json_repair

    if val_pdf_names is None:
        val_pdf_names = []

    train_ds, val_ds = [], []
    seen_images = set()

    for line in open(sft_jsonl_path, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)

        try:
            llm_output = json_repair.loads(rec["output"])
        except Exception:
            llm_output = None

        if not llm_output or rec["image_path"] in seen_images:
            continue
        seen_images.add(rec["image_path"])

        # Task 1: content + structural_elements
        task1_output = {
            "content": llm_output.get("content", {}),
            "structural_elements": llm_output.get("structural_elements", {}),
        }
        # Task 2: everything else
        task2_output = {
            k: v for k, v in llm_output.items()
            if k not in ("content", "structural_elements")
        }

        sft_record = lambda task_prompt, task_output: {
            "conversations": [
                {"from": "human", "value": f"<image>\n{task_prompt}"},
                {"from": "gpt",   "value": json.dumps(task_output, ensure_ascii=False)},
            ],
            "images": [rec["image_path"]],
        }

        records = [
            sft_record(TASK_1_PROMPT, task1_output),
            sft_record(TASK_2_PROMPT, task2_output),
        ]

        is_val = any(v in rec.get("pdf_name", "") for v in val_pdf_names)
        target = val_ds if is_val else train_ds
        target.extend(records)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_path = Path(output_dir) / "train-v1.json"
    val_path   = Path(output_dir) / "val-v1.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_ds, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_ds, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train_ds)} records → {train_path}")
    print(f"Val:   {len(val_ds)} records  → {val_path}")
    return str(train_path), str(val_path)


# ─────────────────────────────────────────────────────────────────────────────
# Print LlamaFactory config
# ─────────────────────────────────────────────────────────────────────────────

def print_finetune_instructions():
    print("=" * 60)
    print("LLAMAFACTORY FINE-TUNING SETUP")
    print("=" * 60)
    print("\n1. Add to LlamaFactory/data/dataset_info.json:\n")
    print(json.dumps(DATASET_INFO, indent=2, ensure_ascii=False))
    print("\n2. Create LlamaFactory/examples/train_lora/ocr_finetune.yaml:\n")
    print(LLAMAFACTORY_YAML)
    print("\n3. Run training:")
    print("   cd LlamaFactory && llamafactory-cli train examples/train_lora/ocr_finetune.yaml")
    print("\n4. Deploy fine-tuned model:")
    print("   - Merge LoRA weights: llamafactory-cli export ...")
    print("   - Upload to S3, serve via vLLM on EC2 g5.xlarge (~$1/hr)")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    # Import the master prompt from ocr_fallback
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.ocr_fallback import MASTER_EXTRACTION_PROMPT

    parser = argparse.ArgumentParser(description="ThakAI Fine-tuning Data Generator")
    parser.add_argument("--pdf-dir",          help="Directory of Arabic PDFs to process")
    parser.add_argument("--output",           default="data/sft", help="Output directory")
    parser.add_argument("--max-pages",        type=int, default=None, help="Max pages per PDF")
    parser.add_argument("--format-llamafactory", action="store_true",
                        help="Convert existing SFT JSONL to LlamaFactory format")
    parser.add_argument("--sft-file",         help="Existing SFT JSONL file to format")
    parser.add_argument("--val-pdfs",         nargs="*", default=[],
                        help="PDF names to use as validation set")
    parser.add_argument("--print-config",     action="store_true",
                        help="Print LlamaFactory configuration and exit")
    args = parser.parse_args()

    if args.print_config:
        print_finetune_instructions()

    elif args.format_llamafactory and args.sft_file:
        format_for_llamafactory(
            sft_jsonl_path=args.sft_file,
            output_dir=args.output,
            val_pdf_names=args.val_pdfs,
        )

    elif args.pdf_dir:
        sft_file = build_sft_dataset(
            pdf_dir=args.pdf_dir,
            output_dir=args.output,
            master_prompt=MASTER_EXTRACTION_PROMPT,
            max_pages_per_pdf=args.max_pages,
        )
        print(f"\nNext: python finetune/build_dataset.py "
              f"--format-llamafactory --sft-file {sft_file}")
    else:
        parser.print_help()
