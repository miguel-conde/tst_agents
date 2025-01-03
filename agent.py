import json
from typing import List, Dict, Callable, Optional, Union

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

from tracer import tracer

# from llm_execenvtool import LLMExecEnvTool
from llm_circular_memory import ChatCircularMemory
from llm_handler import ChatGPTRequester

from abc import ABC#, abstractmethod


# Clase base para los agentes
class BaseAgent(ABC):
    def __init__(
        self, 
        # system_prompt: str, 
        # llm_execenvtool: LLMExecEnvTool, 
        llm_handler: ChatGPTRequester, 
        memory: ChatCircularMemory
        ):
        # self._sys_prompt = system_prompt
        # self._llm_execenvtool = llm_execenvtool
        self._llm_handler = llm_handler
        self._memory = memory
        # self._memory.set_sys_prompt(self._sys_prompt)
        # self._memory.add_message({"role": "developer", "content": self._sys_prompt})
        
        # self._tools_list: List[Dict] = [],
        # self._available_functions: Dict = {}
        
    def set_sys_prompt(self, sys_prompt: str) -> None:
        self._sys_prompt = sys_prompt
        
    def set_tools_list(self, tools_list: List[Dict]) -> None:
        self._tools_list = tools_list
        
    def set_available_functions(self, available_functions: Dict) -> None:
        self._available_functions = available_functions

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
            response = self._llm_handler.request(updated_context, tools=self._tools_list, tool_choice="auto")
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
        
        if function_called not in self._available_functions:
            tracer.error(f"Tool `{function_called}` is not available.")
            return {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_called,
                "content": json.dumps({"error": f"Tool `{function_called}` is not available."}),
            }

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
        
        response = self._llm_handler.request(self._memory.get_all(), tools=self._tools_list, tool_choice="auto")
        
        _ = self._process_response(response)
        
        return self._memory.get_all()[-1].content
    
    
