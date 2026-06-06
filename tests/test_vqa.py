import importlib
import os
import sys
import types
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


def _mock_answer_three_questions_batch(*args, **kwargs):
    questions = kwargs.get("questions") or []
    image_paths = kwargs.get("image_paths") or (args[0] if len(args) > 0 else [])
    if not image_paths:
        return []

    return [
        {
            "image_path": image_paths[0],
            "results": [
                {"question": q, "answer": f"mock answer for {q}", "confidence": 0.95}
                for q in questions
            ]
        }
    ]


def import_vqa_module():
    _ensure_mock_service(
        "app.services.vqa",
        {"answer_three_questions_batch": _mock_answer_three_questions_batch}
    )
    module = importlib.import_module("app.nodes.vqa")
    import app.services.vqa as vqa_service
    vqa_service.answer_three_questions_batch = _mock_answer_three_questions_batch
    return module


def create_dummy_image(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("dummy image content")


def run_case(case_name: str, state: Dict[str, Any]) -> None:
    vqa_module = import_vqa_module()
    result = vqa_module.vqa_node(state)
    print(f"\n=== {case_name} ===")
    print(result)


def main():
    sample_image = "data/sample_vqa_image.jpg"
    create_dummy_image(sample_image)

    cases = [
        {
            "name": "Single image VQA",
            "state": {
                "text": "تحقق من محتوى الصورة.",
                "images": [sample_image],
                "evidence": {}
            }
        },
        {
            "name": "No image input",
            "state": {
                "text": "لا يوجد صور.",
                "images": [],
                "evidence": {}
            }
        }
    ]

    for case in cases:
        run_case(case["name"], case["state"])


if __name__ == "__main__":
    main()
