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
أنت خبير دعم اتخاذ قرار في مؤسسة خيرية. وظيفتك هي توجيه النظام (VQA أو Search) لاستكمال أي نقص، أو التقرير.

البيانات الحالية:
- الطلب: {extracted_text}
- تقرير الأدلة (الجودة/التزييف): {risk_level}
- عدد الصور المرفقة: {image_count}
- العمليات السابقة: {history_summary}
استخدم القواعد التالية للقرار:

1. متى تستخدم VQA (أداة تحليل الصور):
   - إذا كانت هناك صور مرفقة ولكن لم يتم تحديد هويتها.
   - إذا كان المستخدم يطلب دواءً أو مساعدة مالية ورفق صورة "إيصال" أو "روشتة" غير واضحة.
   
2. متى تستخدم Search (أداة البحث):
   - إذا عرفنا اسم الدواء ولكننا لا نعرف سعره أو بدائله.
   - إذا ذكر المستخدم حالة مرضية ولم نجد لها دواءً في قاعدة بياناتنا.
   - إذا كان الطلب عاماً (مثل: "احتاج مساعدة لتصليح السقف") ونحتاج معرفة تكاليف تقديرية.

3. متى تستخدم Report (التقرير النهائي):
   - إذا كان لدينا اسم الدواء، سعره، وحالة المستخدم واضحة.
   - إذا لم تكن هناك صور أو معلومات إضافية ستغير جوهر القرار.

أخرج JSON بهذا التنسيق فقط:
{{
  "next_step": "vqa | search | report",
  "action_details": {{
      "target": "السؤال أو عبارة البحث",
      "reasoning": "لماذا اخترت هذه الأداة بناءً على القواعد أعلاه؟"
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