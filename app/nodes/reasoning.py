from app.services.llm import llm
from app.state import CaseState

def reasoning_node(state: CaseState) -> dict:
    """تحليل الحالة وتحديد الخطوة التالية."""
    text = state.get("text", "")
    history = state.get("inquiry_history", [])
    
    # تحويل التاريخ إلى نص لتجنب خطأ التنسيق
    history_str = "\n".join([f"{item.get('type', 'info')}: {item.get('question', '')}" for item in history])
    
    prompt = f"""
    أنت محلل طبي. بناءً على الطلب التالي وسجل الاستقصاءات السابقة، حدد الخطوة التالية (vqa, search, report).
    
    النص: {text}
    سجل الاستقصاءات:
    {history_str}
    
    أجب بكلمة واحدة فقط: vqa أو search أو report.
    """
    
    decision = llm(prompt).strip().lower()
    
    return {
        "reasoning": {
            "next_step": decision,
            "reasoning_summary": f"تم اتخاذ قرار {decision} بناءً على السياق."
        }
    }
