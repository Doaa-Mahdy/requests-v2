import json
import re
from typing import Any, Dict
from langchain_core.messages import HumanMessage
from app.state import CaseState
from app.services.llm import llm_model

def _safe_parse_json(text: str) -> Dict[str, Any]:
    """Safely extract and parse JSON from LLM response."""
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"\x0c", "", text).strip()
    match = re.search(r"(\{[\s\S]*\})", text)
    if not match:
        return {}
    
    candidate = match.group(1)
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}

def reasoning_node(state: CaseState) -> dict:
    """
    Decision node: determines whether to use VQA, Search, or Report.
    """
    # 1. استخراج وتحويل البيانات إلى نصوص بسيطة لتجنب خطأ التمرير
    case = state.get("normalized_case", {})
    evidence = state.get("evidence", {})
    images = state.get("images", [])
    text = state.get("text", "")
    
    extracted_text = str(case.get('extracted_text', text))
    risk_level = str(evidence.get('cold_acquisition', {}).get('overall_risk_level', 'unknown'))
    image_count = len(images)
    history = state.get("inquiry_history", [])

    history_summary = ""

    if history:
        history_summary = "\n".join(
            f"- {item.get('type')}: {item.get('target','')}"
            for item in history
            if isinstance(item, dict)
        )
    else:
        history_summary = "لا يوجد"
        # 2. بناء نص الـ Prompt
        prompt_text = f"""
    أنت نظام تخطيط جمع معلومات لمؤسسة خيرية إنسانية.

    مهمتك: تحديد أفضل خطوة لجمع المعلومات، مع تحديد "ماذا يجب أن نعرف بالضبط" قبل اتخاذ القرار.

    ━━━━━━━━━━━━━━━━━━━━━━
    📌 المدخلات
    ━━━━━━━━━━━━━━━━━━━━━━
    النص: {extracted_text}
    مستوى الخطر: {risk_level}
    عدد الصور: {image_count}
    سجل العمليات: {history_summary}

    ━━━━━━━━━━━━━━━━━━━━━━
    🧠 الفكرة الأساسية
    ━━━━━━━━━━━━━━━━━━━━━━
    لا تختار أداة فقط.
    بل حدد:
    - ما الذي نحتاج فهمه؟
    - ما الذي إذا عرفناه سيتغير القرار؟
    - ما الفجوة في الفهم؟

    ━━━━━━━━━━━━━━━━━━━━━━
    🧭 متى نستخدم VQA
    ━━━━━━━━━━━━━━━━━━━━━━
    إذا كانت الصورة تحتوي:
    - دليل طبي / إصابة / حريق / مستند
    - أو عنصر بصري غير مفهوم يؤثر على القرار

    → الهدف: فهم محتوى الصورة

    ━━━━━━━━━━━━━━━━━━━━━━
    🧭 متى نستخدم SEARCH (مهم جداً)
    ━━━━━━━━━━━━━━━━━━━━━━
    إذا استخدمت SEARCH، يجب أن تفكر بهذه الطريقة:

    أنت لا تبحث عن كلمة واحدة،
    أنت تبني "خريطة معرفة" تشمل:

    1) تعريف الشيء (what is it?)
    2) خطورته أو معناه (how serious is it?)
    3) تكلفته أو تأثيره (how much does it cost?)
    4) البدائل (alternatives)
    5) السياق الواقعي (real-world context)

    ━━━━━━━━━━━━━━━━━━━━━━
    📌 يجب أن تنتج SEARCH target كالتالي:
    ━━━━━━━━━━━━━━━━━━━━━━
    بدلاً من:
    "house fire cost"

    اكتب:
    - ما معنى الحالة الطبية/الاجتماعية
    - تكلفة الحل أو الإصلاح
    - تقدير الضرر
    - خيارات بديلة

    ━━━━━━━━━━━━━━━━━━━━━━
    🧭 متى نستخدم REPORT
    ━━━━━━━━━━━━━━━━━━━━━━
    فقط إذا:
    - لدينا فهم كافٍ لكل العناصر المهمة
    - لا توجد مصطلحات أو أشياء غير مفهومة
    - ولا توجد فجوات تؤثر على القرار

    ━━━━━━━━━━━━━━━━━━━━━━
    🚨 قاعدة ذهبية
    ━━━━━━━━━━━━━━━━━━━━━━
    إذا كان هناك:
    - مصطلح غير مفهوم
    - أو عنصر قد يغير التقييم
    → يجب SEARCH أو VQA قبل التقرير

    ━━━━━━━━━━━━━━━━━━━━━━
    🟩 أمثلة

    مثال 1:
    "House burned"
    → SEARCH:
    - damage level after house fire
    - reconstruction cost house Egypt
    - temporary shelter options

    مثال 2:
    "medical clot in report"
    → SEARCH:
    - what is blood clot severity
    - treatment options and urgency
    - medication cost and availability

    مثال 3:
    "image of injury"
    → VQA

    مثال 4:
    "clear situation + enough info"
    → REPORT

    ━━━━━━━━━━━━━━━━━━━━━━
    📤 JSON فقط
    ━━━━━━━━━━━━━━━━━━━━━━

    {{
    "next_step": "vqa | search | report",
    "action_details": {{
        "target": [
        "multiple focused queries capturing definition, severity, cost, and options"
        ],
        "reasoning": "why missing knowledge requires expansion before decision"
    }}
    }}
    """
    
    # 3. إرسال الرسالة بشكل صحيح باستخدام HumanMessage
    response = llm_model.invoke([HumanMessage(content=prompt_text)]).content
    
    # 4. معالجة القرار
    decision = _safe_parse_json(response)
    
    next_step = str(decision.get("next_step", "report")).strip().lower()
    action_details = decision.get("action_details", {})
    
    # ضمان أن action_details قاموس قبل استخراج القيم
    if not isinstance(action_details, dict):
        action_details = {}
        
    target = action_details.get("target", "")
    action_reasoning = action_details.get("reasoning", "")
    
    if next_step not in {"vqa", "search", "report"}:
        next_step = "report"
    
    current_loop = state.get("loop_count", 0) + 1

    return {
        "reasoning": {
            "next_step": next_step,
            "question_or_query": target,
            "reasoning": action_reasoning,
        },
        "loop_count": current_loop
    }