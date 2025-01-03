import json
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

from execenvtool import ExecutionEnvironment
from extract_pydocs import extract_documentation

# from memory import ChatMemory
from llm_circular_memory import ChatCircularMemory
from llm_handler import ChatGPTRequester

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
Usa las funciones disponibles para interactuar con el entorno de ejecución.
No uses `print` para mostrar resultados, usa las funciones disponibles.
Para mostrar resultados debes usar siempre la función `format_variable`.
Tus respuestas deben ir en formato markedown.
Cuando se produzcan errores ejecutar el código:
   - Muestra el código que da error
   - Muestra el mensaje de error 
   - Explica por qué no se puede ejecutar 
   - Y sugiere posibles soluciones, pero no tomes iniciativas por tu cuenta para solucionarlo, pide permiso antes.

"""

class LLMExecEnvTool:
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
        self._tools = tools
        self._env = ExecutionEnvironment()
        self._llm_handler = llm_handler
        self._sys_prompt = sys_prompt

        # Import modules dynamically
        import importlib
        for module_name in imports:
            try:
                importlib.import_module(module_name)
                self._env.pyExec(f"import {module_name}")
                self._sys_prompt += f"\n\nEn el módulo `{module_name}`, ya importado, tienes estas funciones:\n{extract_documentation(f'{module_name}.py')}"
            except ModuleNotFoundError as e:
                tracer.error(f"Module `{module_name}` not found: {str(e)}")
                raise ImportError(f"Error importing module `{module_name}`: {str(e)}")

        self._memory = memory
        self._memory.set_sys_prompt(self._sys_prompt)
        self._memory.add_message({"role": "developer", "content": self._sys_prompt})
        
        self._available_functions = {
            "pyExec": self._env.pyExec,
            "format_variable": self._env.format_variable,
            "get_environment": self._env.get_environment,
            "extract_documentation": extract_documentation,
        }

         
    # def _process_response(self, response: ChatCompletionMessage) -> ChatCompletionMessage:
    
    #     if dict(response).get('tool_calls'):

    #         for tool_call in response.tool_calls:
    #             # Which function call was invoked
    #             function_called = tool_call.function.name

    #             # Extracting the arguments
    #             function_args  = json.loads(tool_call.function.arguments)

    #             # Function names
    #             available_functions = {
    #                 "pyExec": self._env.pyExec,
    #                 "format_variable": self._env.format_variable,
    #                 "get_environment": self._env.get_environment,
    #                 "extract_documentation": extract_documentation,
    #             }

    #             function_to_call = available_functions[function_called]
    #             response_message = json.dumps(function_to_call(*list(function_args.values())))
                
    #             tool_response = {
    #                 "tool_call_id": tool_call.id,
    #                 "role": "tool",
    #                 "name": function_called,
    #                 "content": response_message,
    #             }
                
    #         response = self._llm_handler.request(self._memory.get_all() + [response, tool_response], tools=tools, tool_choice="auto")

    #         response = self._process_response(response)

    #     else:
    #         self._memory.add_message(response)

    #     return response
    
    def _process_response(self, response: ChatCompletionMessage) -> ChatCompletionMessage:
        """
        Processes the LLM's response, handling tool calls when necessary.

        If the response includes tool calls, they are executed, and the results are
        added to the context. The LLM is reinvoked with the updated context if needed.

        Args:
            response (ChatCompletionMessage): The LLM's response to process.

        Returns:
            ChatCompletionMessage: The processed response.
        """
        
        if dict(response).get('tool_calls'):
            # Handle tool calls in the LLM's response, if any
            tools_responses = []
            for tool_call in response.tool_calls:
                tools_responses.append(self._handle_tool_call(tool_call))

            # Reinvoke the LLM with updated context containing  tools responses
            updated_context = self._memory.get_all() + [response] + tools_responses
            response = self._llm_handler.request(updated_context, tools=tools, tool_choice="auto")
            return self._process_response(response)

        self._memory.add_message(response)
        return response

    def _handle_tool_call(self, tool_call: ChatCompletionMessageToolCall) -> dict:
        """
        Handles a single tool call from the LLM.

        This method identifies the tool to invoke, executes it with the provided arguments,
        and formats the result as a tool response.

        Args:
            tool_call (ChatCompletionMessageToolCall): Details of the tool call from the LLM.

        Returns:
            dict: The response from the tool, formatted as a message for the LLM.
        """
        function_called = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        try:
            function_to_call = self._available_functions[function_called]
            response_message = json.dumps(function_to_call(**function_args))
        except Exception as e:
            tracer.error(f"Error executing tool `{function_called}` with args `{function_args}`: {str(e)}")
            response_message = json.dumps({"error": f"Tool execution failed: {str(e)}"})

        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_called,
            "content": response_message,
        }

    
    def answer_user_request(self, user_input: str) -> str:
        """
        Handles a user's input and generates a response.

        The user's input is added to memory, and the LLM is queried with the current context.
        Tool calls, if any, are processed recursively.

        Args:
            user_input (str): The user's input to the LLM.

        Returns:
            str: The final response content from the LLM.
        """
        self._memory.add_message({"role": "user", "content": user_input})
        
        response = self._llm_handler.request(self._memory.get_all(), tools=tools, tool_choice="auto")
        
        _ = self._process_response(response)
        
        return self._memory.get_all()[-1].content

# Usage

def main():
    
    from llm_execenvtool import LLMExecEnvTool
    from llm_circular_memory import ChatCircularMemory
    from llm_handler import ChatGPTRequester

    # msgs_memory = ChatMemory(word_limit=20000, n_recent=10)
    llm_handler = ChatGPTRequester()
    msgs_memory = ChatCircularMemory(llm_handler=llm_handler)

    llm_exec = LLMExecEnvTool(llm_handler = llm_handler, memory=msgs_memory, imports = ['joke_cat_dog'])

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