import json
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

from tracer import tracer

from llm_execenvtool import LLMExecEnvTool
from llm_circular_memory import ChatCircularMemory
from llm_handler import ChatGPTRequester

tools = [ {
    'type': 'function',
    'function': {
        "name": "order_python_programmer",
        "description": "Cuando necesites ejecutar algo, este agente LLM, que es un experto programador en Pythob, escribirá el código neesario, lo ejecutará y te informará del resultado",
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
Eres un experto Data Scientist, especializado en EDA (Exploratory Data Analysis). Perteneces a un equipo de científicos de datos que trabajan en modelado predictivo.
Tú eres el responsable de la exploración de los datos. 
Reccibirás indicaciones de tus usarios para que explores datos. 
Pero tú no debes programar NUNCA. Tienes a tu disposición un agente LLM que es un experto programador en Python. Obligatoriamente le utilizarás para que
programe todo lo que tú necesites para llevar a cabo tu trabajo.
"""

class EDAgent:
    def __init__(self, llm_execenvtool: LLMExecEnvTool, llm_handler: ChatGPTRequester, memory: ChatCircularMemory):
        self.llm_execenvtool = llm_execenvtool
        self._llm_handler = llm_handler
        self._sys_prompt = system_prompt
        self._memory = memory
        self._memory.set_sys_prompt(self._sys_prompt)
        self._memory.add_message({"role": "developer", "content": self._sys_prompt})
        
        self._available_functions = {
            "order_python_programmer": self._order_python_programmer,
        }
        
    def _order_python_programmer(self, your_orders: str) -> str:
        return self.llm_execenvtool.answer_user_request(your_orders)

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
    
def main():
    
    from eda_agent import EDAgent
    from llm_execenvtool import LLMExecEnvTool
    from llm_circular_memory import ChatCircularMemory
    from llm_handler import ChatGPTRequester

    llm_handler = ChatGPTRequester()
    msgs_memory = ChatCircularMemory(llm_handler=llm_handler)

    llm_python_programmer = LLMExecEnvTool(llm_handler = llm_handler, memory=msgs_memory, imports = ['joke_cat_dog'])
    llm_eda_agent = EDAgent(llm_execenvtool=llm_python_programmer, llm_handler=llm_handler, memory=msgs_memory)

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