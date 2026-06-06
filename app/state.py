from typing import TypedDict, Any, Dict, List, Optional, Annotated
from langgraph.graph.message import add_messages

class CaseState(TypedDict):
    text: str
    voice_path: Optional[str]
    images: List[str]

    transcript: Optional[str]
    normalized_case: Dict[str, Any]
    evidence: Dict[str, Any]

    inquiry_history: Annotated[List[Dict[str, Any]], add_messages]
    loop_count: int
    
    reasoning: Dict[str, Any]
    final_output: Dict[str, Any]