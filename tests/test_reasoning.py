import importlib
import os
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List

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
    if "needs_followup" in prompt:
        return SimpleNamespace(content='{"needs_followup": false, "followup_questions": [], "analysis": "لا حاجة لمعلومات إضافية."}')

    return SimpleNamespace(content='{"need": "دعم طبي عاجل", "severity": "عالي", "decision": "قبول", "is_emergency": true, "missing_info": [], "reasoning": "الحالة توضح حاجة طبية عاجلة.", "confidence": 0.95}')


def _mock_answer_single_question(image_path: str, question: str, ocr_text: str = "", context: str = "") -> Dict[str, Any]:
    return {
        "question": question,
        "answer": "هذا نص تجريبي للمسألة.",
        "confidence": 0.9,
        "reasoning": "إجابة مُولَّدة بشكل وهمي لأغراض الاختبار."
    }


def import_reasoning_module():
    _ensure_mock_service(
        "app.services.llm",
        {
            "llm": SimpleNamespace(invoke=_mock_llm_invoke),
            "llm_model": SimpleNamespace(invoke=_mock_llm_invoke, bind_tools=lambda tools: SimpleNamespace(invoke=_mock_llm_invoke))
        }
    )
    _ensure_mock_service(
        "app.services.vqa",
        {"answer_single_question": _mock_answer_single_question}
    )
    module = importlib.import_module("app.nodes.reasoning")
    module.llm.invoke = _mock_llm_invoke
    module.answer_single_question = _mock_answer_single_question
    return module


def run_case(case_name: str, state: Dict[str, Any]) -> None:
    reasoning_module = import_reasoning_module()
    result = reasoning_module.reasoning_node(state)
    print(f"\n=== {case_name} ===")
    print(result)


def main():
    cases = [
        {
            "name": "Reasoning with cold evidence and search results",
            "state": {
                "normalized_case": {
                    "extracted_text": "الحالة تشير إلى مريض يحتاج نقل طبي عاجل.",
                    "image_count": 1
                },
                "evidence": {
                    "cold_acquisition": {
                        "overall_risk_score": 0.82,
                        "overall_risk_level": "high"
                    },
                    "vqa_analysis": {
                        "results": [
                            {"question": "ما محتوى الصورة؟", "answer": "وثيقة طبية"}
                        ]
                    }
                },
                "search_results": {"medical_analysis": [{"case_or_drug": "دواء"}]},
                "images": ["data/sample_reasoning_image.jpg"],
                "text": "الشكوى مكتوبة بأن المريض يعاني من ألم حاد ويحتاج دعم عاجل."
            }
        },
        {
            "name": "Reasoning with missing search results",
            "state": {
                "normalized_case": {"extracted_text": "طلب مساعدة ماديّة.", "image_count": 0},
                "evidence": {"cold_acquisition": {"overall_risk_score": 0.35, "overall_risk_level": "medium"}},
                "search_results": {},
                "images": [],
                "text": "يحتاج دعم بسيط لإصلاح منزل.",
            }
        }
    ]

    for case in cases:
        run_case(case["name"], case["state"])


if __name__ == "__main__":
    main()
