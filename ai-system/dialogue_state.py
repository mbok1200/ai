from typing import List, Dict, Any
from pydantic import BaseModel

class DialogueState(BaseModel):
    user_input: str = ""
    user_id: str = ""
    current_node: str
    context: Dict[str, Any] = {}
    collected_data: Dict[str, Any] = {}
    required_fields: List[str] = []
    messages: List[Dict[str, str]] = []
    function_calls: List[Dict[str, Any]] = []
    intent: str = ""
    messages: List[Dict[str, str]] = []