import os
from typing import Dict, Any, List
from app.state import CaseState
from app.services.vqa import answer_three_questions_batch

VQA_QUESTIONS = [
    "ما هو مضمون هذه الصورة؟",
    "هل تحتوي الصورة على نص واضح؟ إذا كان نعم، فما هو؟",
    "هل يوجد أي مؤشر على أن الصورة عبارة عن وثيقة طبية أو إيصال دعم مالي؟"
]

def vqa_node(state: CaseState) -> dict:
    # 1. استخراج الأدلة الحالية أو تهيئتها إذا كانت مفقودة
    evidence = state.get("evidence", {})
    inquiry_history = state.get("inquiry_history", [])
    
    image_paths = state.get("images") or []
    intake_text = state.get("text") or ""
    question = state.get("reasoning", {}).get("question_or_query")

    # 2. الحالة: لا توجد صور - يجب إرجاع evidence حتى لو كانت فارغة
    if not image_paths:
        evidence["vqa_analysis"] = []
        return {
            "evidence": evidence,
            "inquiry_history": inquiry_history
        }

    ocr_texts = [intake_text] * len(image_paths)
    questions = [question] if question else VQA_QUESTIONS

    try:
        vqa_results = answer_three_questions_batch(
            image_paths=image_paths,
            ocr_texts=ocr_texts,
            description="تحليل صور الطلب والتناسق مع نص الشكوى",
            questions=questions
        )
    except Exception as e:
        vqa_results = [{"error": f"VQA processing failed: {str(e)}"}]

    # 3. تحديث الأدلة
    evidence["vqa_analysis"] = {
        "questions": questions,
        "results": vqa_results,
        "metadata": {
            "image_count": len(image_paths),
            "execution_status": "completed"
        }
    }

    inquiry_history = inquiry_history + [{
        "type": "vqa",
        "question": question if question else "default_vqa_questions",
        "results": vqa_results
    }]

    return {
        "evidence": evidence,
        "inquiry_history": inquiry_history
    }