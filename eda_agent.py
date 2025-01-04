
from llm_execenvtool import LLMExecEnvTool
from llm_circular_memory import ChatCircularMemory
from llm_handler import ChatGPTRequester

from agent import BaseAgent

tools = [ {
    'type': 'function',
    'function': {
        "name": "order_python_programmer",
        "description": "Cuando necesites ejecutar algo, este agente LLM, que es un experto programador en Python, escribirá el código neesario, lo ejecutará y te informará del resultado",
        "parameters": {
            "type": "object",
            "properties": {
                "your_orders": {"type": "string", "description": "Tus órdenes para el programador Python"}
            },
            "required": ["your_orders"]
            }
        }
    },
]

system_prompt = """
Eres un experto Data Scientist, especializado en EDA (Exploratory Data Analysis). 
Perteneces a un equipo de científicos de datos que trabajan en modelado predictivo.
Tú eres el responsable de la exploración de los datos. 

Reccibirás indicaciones de tus usarios para que explores datos. 
Pero tú no debes programar NUNCA. Tienes a tu disposición un agente LLM que es un experto programador en Python. 
Obligatoriamente lo utilizarás para que programe todo lo que tú necesites para llevar a cabo tu trabajo.
Cuando la respuesta del programador no te parezca correecta, SIEMPRE deberás explicarle por qué y darle nuevas órdenes.
"""

class EDAgent(BaseAgent):
    
    def __init__(self, llm_execenvtool: LLMExecEnvTool, llm_handler: ChatGPTRequester, memory: ChatCircularMemory):
        super().__init__(llm_handler, memory)
        self.set_sys_prompt(system_prompt)
        self.set_tools_list(tools)
        # self._memory.set_sys_prompt(self._sys_prompt)
        # self._memory.add_message({"role": "developer", "content": self._sys_prompt})
        
        self._llm_execenvtool = llm_execenvtool
        
        self.set_available_functions({
            "order_python_programmer": self._order_python_programmer,
        })
        
    def _order_python_programmer(self, your_orders: str) -> str:
        return self._llm_execenvtool.answer_user_request(your_orders)

# Usage

def main():
    
    from eda_agent import EDAgent

    llm_handler = ChatGPTRequester()
    msgs_memory_programer = ChatCircularMemory(llm_handler=llm_handler)
    msgs_memory_eda_agent = ChatCircularMemory(llm_handler=llm_handler)

    llm_python_programmer = LLMExecEnvTool(llm_handler = llm_handler, memory=msgs_memory_programer, imports = ['joke_cat_dog', 'utils_EDA'])
    llm_eda_agent = EDAgent(llm_execenvtool=llm_python_programmer, llm_handler=llm_handler, memory=msgs_memory_eda_agent)

    # Get user input
    user_input = input("You: ")
    
    # Loop while user input is not exit nor quit 
    while user_input.lower() not in ['exit', 'quit']:
        
        response = llm_eda_agent.answer_user_request(user_input)
        
        print(f"EDA Agent: {response}")
        
        user_input = input("You: ")
        
    tokens_prompt = llm_handler.prompt_tokens
    tokens_completion = llm_handler.completion_tokens
    tokens_total = llm_handler.total_tokens
    
    print(f"Tokens used: {tokens_total} (Prompt: {tokens_prompt}, Completion: {tokens_completion})")

if __name__ == '__main__':
       
    # import environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    main()