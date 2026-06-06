from langchain_core.messages import HumanMessage
from app.state import CaseState
from app.services.llm import llm_model


def report_node(state: CaseState) -> dict:
    evidence = state.get("evidence", {})
    reasoning = state.get("reasoning", {})
    text = state.get("text", "")
    inquiry_history = state.get("inquiry_history", [])

    prompt = f"""
أنت خبير في توزيع المساعدات الإنسانية. مهمتك هي كتابة تقرير نهائي لموظف المؤسسة ليتمكن من صرف المساعدة.

السياق:
- المستخدم مسجل بالفعل (أهلية مستحقة).
- الطلب الحالي هو "أزمة طارئة".

مخرجاتك يجب أن تكون:
1. "تحليل الحاجة": فك تشفير الطلب (ما الذي يحتاجه فعلياً؟).
2. "درجة الطوارئ": صنفها (طارئ جداً/متوسط/عادي).
3. "التوصية الإجرائية":
   - ما الإجراء؟ (صرف مبلغ، شراء دواء، تحويل).
   - ما القيمة/الكمية المقترحة؟
   - لماذا؟ (مبرر مبني على الأدلة).

كن متعاطفاً، عملياً، ومباشراً في خطواتك.

السياق العام:
- النص الأصلي: {{text}}

نتائج الاستدلال:
{{reasoning}}

الأدلة المجمعة:
{{evidence}}

سجل العمليات:
{{inquiry_history}}


أخرج JSON فقط:
{{
  "case_summary": "...",
  "urgent_need": "...",
  "severity": "...",
  "recommended_action": "...",
  "reasoning": "..."
}}
"""

    response = llm_model.invoke([
        HumanMessage(content=prompt)
    ])

    return {
        "final_output": response.content
    }