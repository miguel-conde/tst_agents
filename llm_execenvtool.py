from execenvtool import ExecutionEnvironment
from extract_pydocs import extract_documentation

# from memory import ChatMemory
from llm_circular_memory import ChatCircularMemory
from llm_handler import ChatGPTRequester

from agent import BaseAgent

from tracer import tracer


tools = [ {
    'type': 'function',
    'function': {
        "name": "pyExec",
        "description": "Ejecuta código Python en un entorno persistente.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "El código Python a ejecutar."}
            },
            "required": ["code"]
            }
        }
    },
    {
    'type': 'function',
    'function': {
        "name": "format_variable",
        "description": "Formatea una variable del entorno para hacerla entendible por el LLM.",
        "parameters": {
            "type": "object",
            "properties": {
                "var_name": {"type": "string", "description": "Nombre de la variable a formatear."}
            },
            "required": ["var_name"]
            }
        }
    },
    {
    'type': 'function',
    'function': {
        "name": "get_environment",
        "description": "Devuelve el entorno actual de variables."
        }
    },
    {
    'type': 'function',
    'function': {
        "name": "extract_documentation",
        "description": "Extrae la documentación de un archivo Python (.py) incluyendo docstrings de módulos, clases y funciones.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Ruta del archivo Python."}
            },
            "required": ["file_path"]
            }
        }
    },
    ]

sys_prompt = """
Eres un programador experto en Python. 
Respetas siempre PEP8. 
Tu objetivo es escribir el código necesario y ejecutarlo para realizar la tarea que te pidan. 

Debes saber que ejecutas el código en local y por tanto tienes acceso a todos los recursos de tu máquina.

Usa las tools disponibles para interactuar con el entorno de ejecución.
No uses `print` para mostrar resultados, usa las tools disponibles.
Para mostrar resultados debes usar siempre la tool `format_variable`. 
Esta tool puede truncar el resultado completo si es muy largo. En tal caso, debes indicar explicitamente que el resultado está truncado.

Además conoces la documentación de las funciones de los módulos importados en tu entorno.
Siempre que puedas hacer algo con una función de los módulos importados, debes hacerlo exclusivamente con ella. 
En ese caso, PROHIBIDO escribir código Python directamente. Usarás las funciones de los módulos importados para todo lo que sea posible.

Tus respuestas deben ir en formato markdown.

Cuando se produzcan errores ejecutar el código:
   - Muestra el código que da error
   - Muestra el mensaje de error 
   - Explica por qué no se puede ejecutar 
   - Y sugiere posibles soluciones, pero no tomes iniciativas por tu cuenta para solucionarlo, pide permiso antes.

"""

class LLMExecEnvTool(BaseAgent):
    """
    A tool for managing interactions between a large language model (LLM) and an execution environment.

    This class integrates tools for executing Python code, managing memory, and processing
    LLM responses with tool calls. It dynamically imports specified modules and enriches
    the LLM's prompt with relevant documentation.
    """
    
    def __init__(self, llm_handler: ChatGPTRequester, memory: ChatCircularMemory, imports: list[str] = []):
        """
        Initializes the LLM Execution Environment Tool.

        Args:
            llm_handler (ChatGPTRequester): Handler for interacting with the LLM.
            memory (ChatCircularMemory): Memory for storing interactions.
            imports (list[str], optional): List of module names to import dynamically. Defaults to [].
        """
        super().__init__(llm_handler, memory)
        self.set_tools_list(tools)
        
        self._env = ExecutionEnvironment()

        # Import modules dynamically
        import importlib
        updated_sys_prompt = sys_prompt
        for module_name in imports:
            try:
                importlib.import_module(module_name)
                self._env.pyExec(f"import {module_name}")
                updated_sys_prompt += f"\n\nEn el módulo `{module_name}`, ya importado, tienes estas funciones:\n{extract_documentation(f'{module_name}.py')}"
            except ModuleNotFoundError as e:
                tracer.error(f"Module `{module_name}` not found: {str(e)}")
                raise ImportError(f"Error importing module `{module_name}`: {str(e)}")
        
        self.set_sys_prompt(updated_sys_prompt)    
        # self._memory.set_sys_prompt(self._sys_prompt)
        # self._memory.add_message({"role": "developer", "content": self._sys_prompt})

        self.set_available_functions({
            "pyExec": self._env.pyExec,
            "format_variable": self._env.format_variable,
            "get_environment": self._env.get_environment,
            "extract_documentation": extract_documentation,
        })


# Usage

def main():
    
    from llm_execenvtool import LLMExecEnvTool
    from llm_circular_memory import ChatCircularMemory
    from llm_handler import ChatGPTRequester

    # msgs_memory = ChatMemory(word_limit=20000, n_recent=10)
    llm_handler = ChatGPTRequester()
    msgs_memory = ChatCircularMemory(llm_handler=llm_handler)

    llm_exec = LLMExecEnvTool(llm_handler = llm_handler, memory=msgs_memory, imports = ['joke_cat_dog', 'utils_EDA'])

    # Get user input
    user_input = input("You: ")
    
    # Loop while user input is not exit nor quit 
    while user_input.lower() not in ['exit', 'quit']:
        
        response = llm_exec.answer_user_request(user_input)
        
        print(f"IA: {response}")
        
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