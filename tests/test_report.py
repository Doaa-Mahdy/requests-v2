import importlib
import os
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _ensure_mock_service(module_name: str, functions: Dict[str, Any]) -> types.ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]

    module = types.ModuleType(module_name)
    for name, func in functions.items():
        setattr(module, name, func)
    sys.modules[module_name] = module
    return module


def _mock_llm_invoke(prompt: str):
    return SimpleNamespace(content='{"case_summary": "حالة طبية طارئة بحاجة لرعاية عاجلة.", "urgent_need": "دعم تكاليف علاج عاجل", "severity": "عالي", "key_evidence": "وثيقة طبية وصورة روشتة", "concerns": ["تأكيد المعلومات الطبية","تاريخ المرض غير واضح"], "recommended_action": "قبول الدعم المالي العاجل وتحويل للمستشفى المناسب", "support_options": ["مساعدة مالية كاملة","تحويل لمستشفى / جهة"], "suggested_next_steps": ["التواصل مع وحدة الدعم الطبي","الحصول على تقرير طبي إضافي"], "confidence": 0.9, "admin_summary": "اطلب دعم عاجل مع متابعة طبية"}')


def import_report_module():
    _ensure_mock_service(
        "app.services.llm",
        {
            "llm": SimpleNamespace(invoke=_mock_llm_invoke),
            "llm_model": SimpleNamespace(invoke=_mock_llm_invoke, bind_tools=lambda tools: SimpleNamespace(invoke=_mock_llm_invoke))
        }
    )
    module = importlib.import_module("app.nodes.report")
    module.llm.invoke = _mock_llm_invoke
    return module


def run_case(case_name: str, state: Dict[str, Any]) -> None:
    report_module = import_report_module()
    result = report_module.report_node(state)
    print(f"\n=== {case_name} ===")
    print(result)


def main():
    cases = [
        {
            "name": "Report generation with reasoning output",
            "state": {
                "normalized_case": {"extracted_text": "الشكوى تشير لاحتياج طبي عاجل"},
                "evidence": {
                    "cold_acquisition": {"overall_risk_level": "high"},
                    "vqa_analysis": {"results": [{"question": "ما محتوى الصورة؟", "answer": "روشتة"}]}
                },
                "reasoning": {"final_decision": {"decision": "قبول", "reasoning": "دعم طبي عاجل."}}
            }
        },
        {
            "name": "Report generation with incomplete reasoning",
            "state": {
                "normalized_case": {"extracted_text": "طلب مساعدة مالية.", "image_count": 0},
                "evidence": {},
                "reasoning": {"final_decision": {"decision": "needs review", "reasoning": "معلومات ناقصة."}}
            }
        }
    ]

    for case in cases:
        run_case(case["name"], case["state"])


if __name__ == "__main__":
    main()
