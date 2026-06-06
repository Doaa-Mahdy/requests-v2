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

    # ... داخل دالة vqa_node ...
    
    # 3. تحديث الأدلة (هذا الجزء سليم لأنه يخزن البيانات الخام)
    evidence["vqa_analysis"] = {
        "questions": questions,
        "results": vqa_results,
        "metadata": {"image_count": len(image_paths), "execution_status": "completed"}
    }

    # الحل: تحويل النتائج المعقدة إلى ملخص نصي خفيف قبل إضافتها لـ inquiry_history
    # هذا يمنع تمرير قاموس "results" الضخم إلى الـ LLM في العقد القادمة
    summary_results = []
    for res in vqa_results:
        for r in res.get("results", []):
            summary_results.append(f"Q: {r.get('question')} -> A: {r.get('answer')}")
    
    summary_text = "\n".join(summary_results)

    inquiry_history = inquiry_history + [{
    "type": "vqa",
    "target": question or "general_image_analysis",
    "content": summary_text
}]
    
    return {
        "evidence": evidence,
        "inquiry_history": inquiry_history
    }