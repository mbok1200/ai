from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class DialogueState(BaseModel):
    user_input: str = ""
    user_id: str = ""
    current_node: str = ""
    intent: str = ""
    function_calls: List[Dict] = Field(default_factory=list)
    messages: List[Dict] = Field(default_factory=list)
    context: Any = Field(default_factory=dict)  # Дозволяємо будь-який тип
    collected_data: Dict[str, Any] = {}
    required_fields: List[str] = []