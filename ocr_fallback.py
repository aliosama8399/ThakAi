"""
ThakAI – VLM-First Arabic Extraction
Replaces the original Textract-based OCR fallback.

WHY VLM-FIRST:
  PyMuPDF renders PDF pages as pixel images. For Arabic text, feeding those
  images to a Vision-Language Model (VLM) that natively understands cursive
  Arabic script, diacritics, and mixed layouts is significantly more accurate
  than routing through a classical OCR engine like Amazon Textract.

SUPPORTED BACKENDS (ranked by Arabic quality):
  1. qari_ocr   — QARI-OCR v0.2, fine-tuned Qwen2-VL-2B, best open Arabic OCR
                  (self-hosted on EC2 g5.xlarge via vLLM / SageMaker endpoint)
  2. baseer     — Baseer (Qwen2.5-VL-3B), outputs Markdown, layout-aware
                  (self-hosted on EC2 g5.xlarge via vLLM / SageMaker endpoint)
  3. mistral    — Mistral OCR API (1000 pages/$), good Arabic, no GPU needed
  4. gemini     — Gemini 2.5 Flash (Google), excellent Arabic, API
  5. claude     — Claude 3.5 Sonnet via Amazon Bedrock, excellent Arabic, AWS-native
  6. qwen_vl    — Qwen2.5-VL-7B on SageMaker JumpStart (managed, open-source)

Amazon Textract is retained as a last-resort fallback only for cases where all
VLM backends are unavailable.
"""
import base64
import json
import os
from enum import Enum

import boto3
from loguru import logger

from config.settings import settings


class OCRBackend(str, Enum):
    QARI_OCR = "qari_ocr"       # self-hosted, best Arabic accuracy
    BASEER = "baseer"            # self-hosted, Markdown output
    MISTRAL = "mistral"          # API, 1000 pages/$
    GEMINI = "gemini"            # Google API, excellent Arabic
    CLAUDE = "claude"            # Amazon Bedrock, AWS-native
    QWEN_VL = "qwen_vl"         # SageMaker JumpStart
    TEXTRACT = "textract"        # classical fallback — last resort


_ARABIC_EXTRACTION_PROMPT = (
    "You are an Arabic document digitization specialist. "
    "Extract ALL text from this document image exactly as written. "
    "Preserve the original Arabic text, numbers, punctuation, and document structure. "
    "Output the extracted text only — no explanations, no translations."
)

_BILINGUAL_EXTRACTION_PROMPT = (
    "Extract ALL text from this document image exactly as written. "
    "The document may contain Arabic and/or English text. "
    "Preserve the original text, numbers, punctuation, and structure. "
    "Output the extracted text only."
)


# ─────────────────────────────────────────────────────────────────────────────
# Primary entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_arabic_from_images(
    page_images: list[bytes],
    backend: str = OCRBackend.CLAUDE,
    language_hint: str = "arabic",
) -> list[str]:
    """
    Extract Arabic (or bilingual) text from rendered PDF page images using a VLM.

    Parameters
    ----------
    page_images : list[bytes]
        PNG bytes per page from ingestion/text_extraction.extract_page_images().
    backend : str
        Which VLM backend to use. See OCRBackend enum.
    language_hint : str
        Hint passed to the model prompt.

    Returns
    -------
    list[str]
        Extracted text per page, in order.
    """
    results = []
    backend = OCRBackend(backend)
    logger.info(f"VLM extraction: {len(page_images)} pages via backend={backend.value}")

    for i, img_bytes in enumerate(page_images):
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        try:
            if backend == OCRBackend.QARI_OCR:
                text = _extract_qari_ocr(b64)
            elif backend == OCRBackend.BASEER:
                text = _extract_baseer(b64)
            elif backend == OCRBackend.MISTRAL:
                text = _extract_mistral_ocr(b64)
            elif backend == OCRBackend.GEMINI:
                text = _extract_gemini(b64)
            elif backend == OCRBackend.CLAUDE:
                text = _extract_claude_bedrock(b64, language_hint)
            elif backend == OCRBackend.QWEN_VL:
                text = _extract_qwen_sagemaker(b64)
            elif backend == OCRBackend.TEXTRACT:
                logger.warning("Using Amazon Textract (classical OCR). Not recommended for Arabic.")
                text = _extract_textract_bytes(img_bytes)
            else:
                raise ValueError(f"Unknown backend: {backend}")
        except Exception as exc:
            logger.error(f"VLM extraction failed page {i+1} ({backend.value}): {exc}")
            text = ""

        results.append(text)
        logger.debug(f"  Page {i+1}: extracted {len(text)} chars")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Backend implementations
# ─────────────────────────────────────────────────────────────────────────────

def _extract_claude_bedrock(b64_image: str, language_hint: str = "arabic") -> str:
    """Claude 3.5 Sonnet via Amazon Bedrock. AWS-native, no extra infrastructure."""
    bedrock = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    prompt = _ARABIC_EXTRACTION_PROMPT if language_hint == "arabic" else _BILINGUAL_EXTRACTION_PROMPT

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64_image},
                },
                {"type": "text", "text": prompt},
            ],
        }],
    })

    response = bedrock.invoke_model(
        modelId=settings.bedrock_model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def _extract_mistral_ocr(b64_image: str) -> str:
    """
    Mistral OCR API — 1000 pages/$ (batch: ~2000 pages/$).
    Good Arabic support, table-aware, no GPU needed.
    Set MISTRAL_API_KEY in environment.
    """
    import urllib.request

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY environment variable not set.")

    payload = json.dumps({
        "model": "mistral-ocr-latest",
        "document": {"type": "image_url", "image_url": f"data:image/png;base64,{b64_image}"},
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.mistral.ai/v1/ocr",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    return "\n\n".join(p.get("markdown", "") for p in data.get("pages", []))


def _extract_gemini(b64_image: str) -> str:
    """
    Gemini 2.5 Flash via Google AI API. Excellent Arabic quality.
    Set GOOGLE_API_KEY in environment.
    """
    import urllib.request

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set.")

    model = "gemini-2.5-flash-preview-05-20"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = json.dumps({
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": b64_image}},
                {"text": _ARABIC_EXTRACTION_PROMPT},
            ]
        }]
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _extract_qari_ocr(b64_image: str) -> str:
    """
    QARI-OCR v0.2 — best open-source Arabic OCR.
    Fine-tuned Qwen2-VL-2B. CER=0.061, WER=0.160, BLEU=0.737.
    Model: https://huggingface.co/riotu-lab/QARI-OCR

    Deployment: EC2 g5.xlarge + vLLM OR Amazon SageMaker endpoint.
    Set QARI_OCR_ENDPOINT (vLLM URL) or QARI_OCR_SAGEMAKER_ENDPOINT.
    """
    endpoint_url = os.environ.get("QARI_OCR_ENDPOINT", "")
    if endpoint_url:
        return _call_openai_compatible_vlm(
            endpoint_url, "QARI-OCR", b64_image,
            "Extract the Arabic text from this document image.",
            api_key=os.environ.get("QARI_OCR_API_KEY", "none"),
        )
    sm_endpoint = os.environ.get("QARI_OCR_SAGEMAKER_ENDPOINT", "")
    if sm_endpoint:
        return _call_sagemaker_vlm_endpoint(sm_endpoint, b64_image)
    raise RuntimeError("QARI-OCR not configured. Set QARI_OCR_ENDPOINT or QARI_OCR_SAGEMAKER_ENDPOINT.")


def _extract_baseer(b64_image: str) -> str:
    """
    Baseer — Arabic document to Markdown.
    Fine-tuned Qwen2.5-VL-3B on 500k Arabic document image pairs.
    Model: https://huggingface.co/Misraj/Baseer

    Set BASEER_ENDPOINT (vLLM URL) or BASEER_SAGEMAKER_ENDPOINT.
    """
    endpoint_url = os.environ.get("BASEER_ENDPOINT", "")
    if endpoint_url:
        return _call_openai_compatible_vlm(
            endpoint_url, "Baseer", b64_image,
            "Convert this Arabic document image to structured Markdown.",
            api_key=os.environ.get("BASEER_API_KEY", "none"),
        )
    sm_endpoint = os.environ.get("BASEER_SAGEMAKER_ENDPOINT", "")
    if sm_endpoint:
        return _call_sagemaker_vlm_endpoint(sm_endpoint, b64_image)
    raise RuntimeError("Baseer not configured. Set BASEER_ENDPOINT or BASEER_SAGEMAKER_ENDPOINT.")


def _extract_qwen_sagemaker(b64_image: str) -> str:
    """
    Qwen2.5-VL via Amazon SageMaker JumpStart.
    Available in SageMaker model catalog; excellent Arabic document OCR.
    Set QWEN_VL_SAGEMAKER_ENDPOINT in environment.
    """
    sm_endpoint = os.environ.get("QWEN_VL_SAGEMAKER_ENDPOINT", "")
    if not sm_endpoint:
        raise RuntimeError("QWEN_VL_SAGEMAKER_ENDPOINT environment variable not set.")
    return _call_sagemaker_vlm_endpoint(sm_endpoint, b64_image)


# ─────────────────────────────────────────────────────────────────────────────
# Generic endpoint callers
# ─────────────────────────────────────────────────────────────────────────────

def _call_openai_compatible_vlm(
    endpoint_url: str, model_name: str, b64_image: str,
    prompt: str, api_key: str = "none",
) -> str:
    """Call any vLLM / OpenAI-compatible VLM endpoint."""
    import urllib.request

    payload = json.dumps({
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": 4096,
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{endpoint_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def _call_sagemaker_vlm_endpoint(endpoint_name: str, b64_image: str) -> str:
    """Call a SageMaker real-time endpoint serving a VLM via TGI or vLLM."""
    sm_runtime = boto3.client("sagemaker-runtime", region_name=settings.aws_region)
    payload = json.dumps({
        "inputs": _ARABIC_EXTRACTION_PROMPT,
        "image": b64_image,
        "parameters": {"max_new_tokens": 2048, "do_sample": False},
    })
    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    result = json.loads(response["Body"].read())
    if isinstance(result, list):
        return result[0].get("generated_text", "")
    return result.get("generated_text", "")


# ─────────────────────────────────────────────────────────────────────────────
# Classical Textract — last resort only
# ─────────────────────────────────────────────────────────────────────────────

def _extract_textract_bytes(image_bytes: bytes) -> str:
    """
    Amazon Textract on a single page image — LAST RESORT only.
    Not recommended for Arabic; use extract_arabic_from_images() instead.
    """
    textract = boto3.client("textract", region_name=settings.aws_region)
    response = textract.detect_document_text(Document={"Bytes": image_bytes})
    lines = [
        block["Text"]
        for block in response.get("Blocks", [])
        if block["BlockType"] == "LINE"
    ]
    return "\n".join(lines)
