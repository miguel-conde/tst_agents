from langchain import hub
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from typing import List, Dict, Any
import json

from execenvtool import ExecutionEnvironment
from extract_pydocs import extract_documentation
from tracer import tracer



sys_prompt = """
Eres un programador experto en Python. 
Respetas siempre PEP8. 
Tu objetivo es escribir el código necesario y ejecutarlo para realizar la tarea que te pidan. 

Debes saber que ejecutas el código en local y por tanto tienes acceso a todos los recursos de tu máquina.

Para ejecutar código python en local tienes la tool `pyExec`. 
Todas las variables que crees en el código se guardarán en el entorno de ejecución y podrás usarlas en llamadas posteriores.
Siempre debes devolver resultados explícitamente.
Para devolver resultados debes usar siempre la tool `format_variable`. No uses `print`.
Esta tool puede truncar el resultado completo si es muy largo. En tal caso, debes indicar explicitamente que el resultado está truncado.

Para obtener el entorno actual de variables usa la tool `get_environment`.

Además conoces la documentación de las funciones de los módulos importados en tu entorno.
Siempre que puedas hacer algo con una tool o con una función de los módulos importados, debes hacerlo exclusivamente con ella. 
En ese caso, PROHIBIDO escribir código Python directamente. Usarás las funciones de los módulos importados para todo lo que sea posible.

Tus respuestas deben ir en formato markdown.

Cuando se produzcan errores ejecutar el código:
   - Muestra el código que da error
   - Muestra el mensaje de error 
   - Explica por qué no se puede ejecutar 
   - Y sugiere posibles soluciones, pero no tomes iniciativas por tu cuenta para solucionarlo, pide permiso antes.

"""


class LCAgent:

    _env = ExecutionEnvironment()

    @staticmethod
    @tool
    def pyExec(code: str) -> str:
        """
        Ejecuta el código Python dado utilizando el entorno interno.
        
        Parameters:
        code (str): Código Python a ejecutar.

        Returns:
        str: Mensaje con las variables en el entorno tras la ejecución.
        """
        return LCAgent._env.pyExec(code)

    @staticmethod
    @tool
    def format_variable(var_name: str):
        """
        Convierte una variable del entorno a un formato entendible por el LLM.
        Si la variable es demasiado grande, la convierte parcialmente.

        Parameters:
        var_name (str): Nombre de la variable a formatear.

        Returns:
        dict: Representación formateada de la variable o un error si no existe.
        """
        return LCAgent._env.format_variable(var_name)

    @staticmethod
    @tool
    def get_environment() -> str:
        """
        Devuelve el entorno actual de variables.
        """
        return LCAgent._env.get_environment()

    @staticmethod
    @tool
    def extract_doc(file_path: str) -> Dict[str, Any]:
        """
        Extrae la documentación de un archivo Python (.py) incluyendo docstrings de módulos, clases y funciones.

        Args:
            file_path (str): Ruta del archivo Python.

        Returns:
            Dict[str, Any]: Un diccionario con las docstrings organizadas por tipo (module, classes, functions).
        """
        return extract_documentation(file_path)


    def __init__(self, imports: List[str] = []):

        self._tools = [LCAgent.pyExec, LCAgent.format_variable, LCAgent.get_environment, LCAgent.extract_doc]

        # Import modules dynamically
        import importlib
        updated_sys_prompt = sys_prompt
        for module_name in imports:
            try:
                importlib.import_module(module_name)
                self._env.pyExec(f"import {module_name}")
                extracted_doc = extract_documentation(f'{module_name}.py')
                updated_sys_prompt += f"\n\nEn el módulo `{module_name}`, ya importado, tienes estas funciones:\n{extracted_doc}"
            except ModuleNotFoundError as e:
                tracer.error(f"Module `{module_name}` not found: {str(e)}")
                raise ImportError(f"Error importing module `{module_name}`: {str(e)}")
        
        self._prompt = ChatPromptTemplate.from_messages(
            [
                # ("system", updated_sys_prompt),
                ("system", sys_prompt),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self._chat = ChatOpenAI(model="gpt-4o", temperature=0.0)
        # self._chain = self._prompt | self.chat

        self._agent = create_tool_calling_agent(self._chat, self._tools, self._prompt)
        self._executor = AgentExecutor(agent = self._agent, tools = self._tools)
        
        self.history = ChatMessageHistory()

    def answer_user_request(self, user_request: str) -> str:
        
        self.history.add_message(HumanMessage(content=user_request))
        # response = self._chain.invoke({"messages": self.history.messages})
        response = self._executor.invoke({"messages": self.history.messages})
        self.history.add_message(AIMessage(content=response['output']))

        return response['output']
    

# Usage
def main():
    
    agent = LCAgent(imports=['joke_cat_dog', 'utils_EDA'])
    
    user_input = input("You: ")
    
    while user_input.lower() not in ['exit', 'quit']:
        
        response = agent.answer_user_request(user_input)
        
        print(f"AI: {response}")
        
        user_input = input("You: ")
        
    print("Goodbye!")

if __name__ == '__main__':
    main()