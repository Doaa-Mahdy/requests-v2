from typing import TypedDict, Any, Dict, List, Optional


class CaseState(TypedDict):
    # input
    text: str
    voice_path: Optional[str]
    images: List[str]

    # intake output
    transcript: Optional[str]          # Parsed audio transcript data
    normalized_case: Dict[str, Any]

    # evidence layer
    evidence: Dict[str, Any]

    # reasoning
    reasoning: Dict[str, Any]

    # search
    search_query: Optional[str]
    search_results: Optional[str]

    # final
    final_output: Dict[str, Any]