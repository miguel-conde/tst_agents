import openai
import json

from execenvtool import ExecutionEnvironment
from extract_pydocs import extract_documentation
import joke_cat_dog


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
    
    def __init__(self, imports = []):
         self._tools = tools
         self._env = ExecutionEnvironment()
         
         self._client = openai.OpenAI()
         
         self._sys_prompt = sys_prompt
         
         for x in imports:
            # exec(f"import {x}")
            self._env.pyExec(f"import {x}")
            self._sys_prompt += f"\n\nEn el módulo `{x}`, que ya está importado, tienes estas funciones que deberás usar siempre que sea pertinente:{extract_documentation(f'{x}.py')}"

         self._messages = [
             {"role": "system", "content": self._sys_prompt},
             ]
         
    def _process_response(self, response):
        response_message = response.choices[0].message
        self._messages.append(response_message)
    
        if dict(response_message).get('tool_calls'):

            for tool_call in response_message.tool_calls:
                # Which function call was invoked
                function_called = tool_call.function.name

                # Extracting the arguments
                function_args  = json.loads(tool_call.function.arguments)

                # Function names
                available_functions = {
                    "pyExec": self._env.pyExec,
                    "format_variable": self._env.format_variable,
                    "get_environment": self._env.get_environment,
                    "extract_documentation": extract_documentation,
                }

                fuction_to_call = available_functions[function_called]
                response_message = json.dumps(fuction_to_call(*list(function_args .values())))

                self._messages.append({"tool_call_id": tool_call.id,
                                       "role": "tool",
                                       "name": function_called,
                                       "content": response_message,
                                        })

            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=self._messages,
                tools=tools,
                tool_choice="auto"
                )

            response = self._process_response(response)

        else:
            response_message = response_message.content
            self._messages.append({"role": "assistant",
                                    "content": response_message,
                                     })

        return response
    
    def answer_user_request(self, user_input):
        self._messages.append({"role": "user", "content": user_input})
        
        response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=self._messages,
            tools=self._tools,
            tool_choice="auto"
            )
        
        _ = self._process_response(response)
        
        return self._messages[-1]['content']
