# Create a skeleton for a main.py file
import os
import sys

import openai
import json

from execenvtool import ExecutionEnvironment

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
    ]

# Instanciar el entorno
env = ExecutionEnvironment()

client =openai.OpenAI()

messages = []

def process_response(response):
    response_message = response.choices[0].message
    messages.append(response_message)
    
    if dict(response_message).get('tool_calls'):
        
        for tool_call in response_message.tool_calls:
            # Which function call was invoked
            function_called = tool_call.function.name
            
            # Extracting the arguments
            function_args  = json.loads(tool_call.function.arguments)
        
            # Function names
            available_functions = {
                "pyExec": env.pyExec,
                "format_variable": env.format_variable,
                "get_environment": env.get_environment,
            }
            
            fuction_to_call = available_functions[function_called]
            response_message = json.dumps(fuction_to_call(*list(function_args .values())))
            
            messages.append({"tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_called,
                            "content": response_message,
                            })
            
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
            )
        
        response = process_response(response)
        
    else:
        response_message = response_message.content
        messages.append({"role": "assistant",
                         "content": response_message,
                         })
        
    return response

sys_prompt = """
Eres un programador experto en Python. 
Respetas siempre PEP8. 
Tu objetivo es escribir el código necesario y ejecutarlo para realizar la tarea que te pidan. 
Tus respuestas deben ir en formato markedown.
Cuando se produzcan errores ejecutar el código:
   - Muestra el código que da error
   - Muestra el mensaje de error 
   - Explica por qué no se puede ejecutar 
   - Y sugiere posibles soluciones, pero no tomes iniciativas por tu cuenta para solucionarlo, pide permiso antes.
"""

messages = [
    {"role": "system", "content": sys_prompt},
]


def main():
    # Get user input
    user_input = input("You: ")
    
    # Loop while user input is not exit nor quit 
    while user_input.lower() not in ['exit', 'quit']:
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
            )
        
        process_response(response)
        
        print(messages[-1]['content'])
        
        user_input = input("You: ")

if __name__ == '__main__':
    main()