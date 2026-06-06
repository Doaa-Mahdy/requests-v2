import json
import os
import re
from typing import Any, Dict, List

from app.services.llm import llm
from app.services.vqa import answer_single_question


def _clean_json_text(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"\x0c", "", text)
    return text.strip()


def _safe_parse_json(text: str) -> Dict[str, Any]:
    text = _clean_json_text(text)
    match = re.search(r"(\{[\s\S]*\})", text)
    if not match:
        return {}

    candidate = match.group(1)
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


def _parse_followup_questions(text: str) -> List[str]:
    text = _clean_json_text(text)
    parsed = _safe_parse_json(text)
    if isinstance(parsed.get("followup_questions"), list):
        return [str(item).strip() for item in parsed["followup_questions"] if str(item).strip()]

    questions: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        question_match = re.search(r"(?:Question\s*\d*[:\-]\s*)(.+)", line)
        if question_match:
            questions.append(question_match.group(1).strip())
            continue
        if line.endswith("?"):
            questions.append(line)
    return questions


def _load_search_results(search_results: Any) -> Dict[str, Any]:
    if isinstance(search_results, str):
        try:
            return json.loads(search_results)
        except json.JSONDecodeError:
            return {"raw": search_results}
    if isinstance(search_results, dict):
        return search_results
    return {}


def _build_summary(state: Dict[str, Any]) -> str:
    evidence = state.get("evidence", {})
    normalized = state.get("normalized_case", {})
    search = _load_search_results(state.get("search_results", {}))
    summary = {
        "normalized_case": normalized,
        "cold_evidence": evidence.get("cold_acquisition", {}),
        "vqa_analysis": evidence.get("vqa_analysis", {}),
        "search_results": search
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def _query_vqa(image_path: str, question: str, ocr_text: str = "") -> Dict[str, Any]:
    try:
        answer = answer_single_question(image_path=image_path, question=question, ocr_text=ocr_text)
        return {
            "image_path": image_path,
            "question": question,
            "answer": answer.get("answer", ""),
            "confidence": answer.get("confidence", 0.0),
            "reasoning": answer.get("reasoning", "")
        }
    except Exception as e:
        return {
            "image_path": image_path,
            "question": question,
            "answer": "",
            "confidence": 0.0,
            "reasoning": f"VQA failed: {str(e)}"
        }


def reasoning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    evidence_summary = _build_summary(state)
    image_paths = state.get("images") or []
    intake_text = state.get("text", "")

    followup_prompt = f"""
أنت محقق داخلي للحالة.

راجع الأدلة الأولية التالية والمعلومات المتاحة:

{evidence_summary}

أجب فقط بصيغة JSON.

ما إذا كان هناك نقص في المعلومات يمكن سدّه من خلال أسئلة VQA على الصورة.
إذا كان هناك نقص، أعطِ قائمة قصيرة من أسئلة VQA مفيدة لحل الغموض.

الصيغة المطلوبة:
{{
  "needs_followup": true/false,
  "followup_questions": ["..."],
  "analysis": "موجز سبب الحاجة أو عدم الحاجة لمعلومات إضافية"
}}
"""

    followup_response = llm.invoke(followup_prompt).content
    followup_data = _safe_parse_json(followup_response)
    followup_questions = _parse_followup_questions(followup_response)
    needs_followup = bool(followup_data.get("needs_followup", False) or followup_questions)

    inquiry_history: List[Dict[str, Any]] = []
    if needs_followup and followup_questions and image_paths:
        for question in followup_questions:
            for image_path in image_paths:
                if os.path.exists(image_path):
                    inquiry_history.append(_query_vqa(image_path, question, intake_text))

    evidence_with_inquiry = state.get("evidence", {})
    if inquiry_history:
        evidence_with_inquiry["active_inquiry"] = {
            "questions": followup_questions,
            "vqa_results": inquiry_history
        }

    final_prompt = f"""
أنت نظام استنتاج نهائي داخل مؤسسة خيرية.

استخدم المعلومات التالية فقط:
- الحالة المعيارية
- الأدلة الباردة من مرحلة الاكتساب
- نتائج البحث
- نتائج أية استفسارات VQA

موجز المعلومات:
{evidence_summary}

التحقيق الإضافي:
{json.dumps(inquiry_history, ensure_ascii=False, indent=2)}

أنتج فقط JSON نهائي:
{{
  "need": "وصف مختصر للاحتياج الطارئ",
  "severity": "منخفض | متوسط | عالي | طارئ",
  "decision": "قبول | رفض | يحتاج مراجعة",
  "is_emergency": true/false,
  "missing_info": ["..."],
  "reasoning": "سبب القرار بناءً على الأدلة والتحقيق",
  "confidence": 0.0
}}
"""

    final_response = llm.invoke(final_prompt).content
    final_decision = _safe_parse_json(final_response)
    if not final_decision:
        final_decision = {
            "need": "",
            "severity": "",
            "decision": "needs review",
            "is_emergency": False,
            "missing_info": [],
            "reasoning": final_response.strip(),
            "confidence": 0.0
        }

    return {
        "reasoning": {
            "final_decision": final_decision,
            "inquiry_history": inquiry_history,
            "followup_questions": followup_questions,
            "summary": evidence_summary
        },
        "evidence": evidence_with_inquiry
    }
