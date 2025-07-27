import json
from langgraph.graph import StateGraph
from dialogue_state import DialogueState
from tools.config.functions import get_functions
from tools.redmine_api import RedmineAPI

class Workflow:
    def __init__(self, openai_client=None):
        self.workflow = StateGraph(DialogueState)
        self.state = DialogueState(current_node="access_to_redmine")
        self.redmine_api = RedmineAPI()
        self.openai_client = openai_client
        self._setup_navigation_flow()
    def _setup_navigation_flow(self):
        # Define nodes
        self.workflow.add_node("analyze_intent", self.analyze_intent)
        self.workflow.add_node("execute_function", self.execute_function)
        self.workflow.add_node("generate_response", self.generate_response)


        self.workflow.add_node("access_to_redmine", self.redmine_api.access_to_redmine)
        self.workflow.add_node("get_issue_by_date", self.redmine_api.get_issue_by_date)
        self.workflow.add_node("get_issue_by_id", self.redmine_api.get_issue_by_id)
        self.workflow.add_node("get_issue_by_name", self.redmine_api.get_issue_by_name)
        self.workflow.add_node("get_issue_hours", self.redmine_api.get_issue_hours)
        self.workflow.add_node("fill_issue_hours", self.redmine_api.fill_issue_hours)
        self.workflow.add_node("get_user_status", self.redmine_api.get_user_status)
        self.workflow.add_node("set_user_status", self.redmine_api.set_user_status)
        self.workflow.add_node("create_issue", self.redmine_api.create_issue)
        self.workflow.add_node("assign_issue", self.redmine_api.assign_issue)
        self.workflow.add_node("get_wiki_info", self.redmine_api.get_wiki_info)
        # Define edges
        self.workflow.add_edge("analyze_intent", "execute_function")
        self.workflow.add_edge("execute_function", "generate_response")
        # Define edges and conditions
        self.workflow.add_edge("get_issue_by_date", "access_to_redmine")
        self.workflow.add_edge("get_issue_by_id", "access_to_redmine")
        self.workflow.add_edge("get_issue_by_name", "access_to_redmine")
        self.workflow.add_edge("get_issue_hours", "access_to_redmine")

        # Set entry point
        self.workflow.set_entry_point("analyze_intent")
        
        self.app = self.workflow.compile()
    def process_user_input(self, user_input: str, user_id: str = "") -> str:
        """Основний метод для обробки запиту користувача"""
        
        # Ініціалізуємо стан
        initial_state = DialogueState(
            user_input=user_input,
            user_id=user_id,
            current_node="analyze_intent"
        )
        
        # Виконуємо workflow
        final_state = self.app.invoke(initial_state)
        
        # Повертаємо останню відповідь асистента
        if final_state.messages:
            return final_state.messages[-1]["content"]
        else:
            return "Вибачте, не вдалося обробити ваш запит."
    def execute_function(self, state: DialogueState) -> DialogueState:
        """Виконує відповідну функцію на основі аналізу наміру"""
        
        if not state.function_calls:
            return state
            
        function_call = state.function_calls[0]
        function_name = function_call["name"]
        arguments = function_call["arguments"]
        
        # Викликаємо відповідну функцію RedmineAPI
        if hasattr(self.redmine_api, function_name):
            func = getattr(self.redmine_api, function_name)
            try:
                # Підготовка аргументів
                if arguments:
                    # Конвертуємо аргументи у формат, очікуваний API
                    args = [arguments.get(f"value_{i+1}") for i in range(len(arguments))]
                    result = func(state, *args)
                else:
                    result = func(state)
                    
                state = result
                state.current_node = "generate_response"
                
            except Exception as e:
                print(f"Помилка виконання функції {function_name}: {e}")
                state.current_node = "handle_error"
        
        return state

    def generate_response(self, state: DialogueState) -> DialogueState:
        """Генерує відповідь користувачу за допомогою OpenAI"""
        
        context = f"""
        Користувач запитав: {state.user_input}
        Виконана функція: {state.intent}
        Результат: {state.context}
        
        Сформуй природну відповідь українською мовою.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a navigation assistant. Analyze user requests and determine what functions to call.
                        If you have enough information, call the appropriate navigation functions.
                        If information is missing, use request_missing_info function."""
                    },
                    {"role": "user", "content": context}
                ],
                max_tokens=300
            )
            
            ai_response = response.choices[0].message.content
            state.messages.append({
                "role": "assistant",
                "content": ai_response
            })
            
        except Exception as e:
            print(f"Помилка генерації відповіді: {e}")
            state.messages.append({
                "role": "assistant",
                "content": "Вибачте, сталася помилка при обробці вашого запиту."
            })
        
        return state
    def analyze_intent(self, state: DialogueState) -> DialogueState:
        """Аналізує намір користувача за допомогою OpenAI"""
        
        # Функції для OpenAI function calling
        functions = get_functions()
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Ти аналізуєш запити користувачів для системи управління завданнями Redmine. Визнач яку функцію потрібно викликати на основі запиту користувача.Також, якщо потрібно, запитай додаткову інформацію у користувача."
                    },
                    {
                        "role": "user",
                        "content": state.user_input
                    }
                ],
                functions=functions,
                function_call="auto"
            )
            
            message = response.choices[0].message
            
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                state.intent = function_name
                state.function_calls = [{
                    "name": function_name,
                    "arguments": function_args
                }]
                state.current_node = function_name
            else:
                state.current_node = "handle_general_query"
                
        except Exception as e:
            print(f"Помилка при аналізі наміру: {e}")
            state.current_node = "handle_error"
        
        return state
   