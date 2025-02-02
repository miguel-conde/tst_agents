from typing import List, Dict, Callable, Optional, Union
# from pydantic import BaseModel
# from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from circular_memory import CircularMemoryWithBuffer
from llm_handler import ConversationMessage, ChatGPTRequester

class ChatCircularMemory:
    """
    Manages a hierarchical memory system for a chatbot using circular memories for short-term and mid-term memories.

    Attributes:
        sys_prompt (str): The system prompt to initialize the conversation.
        long_term_memory (List[ConversationMessage]): The list storing long-term memory.
        mid_term_memory (CircularMemoryWithBuffer): Circular memory for mid-term storage and summarization.
        short_term_memory (CircularMemoryWithBuffer): Circular memory for short-term storage.
        client (OpenAI): The OpenAI client for generating summaries and interacting with the model.
    """

    def __init__(
        self, 
        sys_prompt: str = None,
        llm_handler: ChatGPTRequester = None,
        short_term_memory_size: int = 10,
        short_term_buffer_size: int = 5,
        mid_term_memory_size: int = 5,
        mid_term_buffer_size: int = 5,
        ) -> None:
        """
        Initializes the ChatCircularMemory instance.

        Args:
            sys_prompt (str): The system prompt to initialize the chatbot's long-term memory.
            short_term_memory_size (int): The size of the short-term memory.
            short_term_buffer_size (int): The size of the short-term buffer.
            mid_term_memory_size (int): The size of the mid-term memory.
            mid_term_buffer_size (int): The size of the mid-term buffer.
        """
        if sys_prompt is None:
            self.sys_prompt = "Eres un asistente amable. Tu nombre es Aya y tienes 49 años."
        else:
            self.sys_prompt = sys_prompt
        self.short_term_memory_size = short_term_memory_size
        self.short_term_buffer_size = short_term_buffer_size
        self.mid_term_memory_size = mid_term_memory_size
        self.mid_term_buffer_size = mid_term_buffer_size
        self.long_term_memory: List[ConversationMessage] = [ConversationMessage(role="system", content=self.sys_prompt)]
        self.mid_term_memory = CircularMemoryWithBuffer(
            size = mid_term_memory_size, 
            buffer_size=mid_term_buffer_size, 
            summarize_fn=self.summary_lt_memory, 
            summarize_fn_args=[self.long_term_memory],
            )
        self.short_term_memory = CircularMemoryWithBuffer(
            size = short_term_memory_size, 
            buffer_size=short_term_buffer_size, 
            summarize_fn=self.summary_buffer, 
            summarize_fn_args=[self.mid_term_memory],
            )
        # self.client = OpenAI()
        if llm_handler is None:
            self.llm_handler = ChatGPTRequester()
        else:
            self.llm_handler = llm_handler
            
    def set_sys_prompt(self, sys_prompt: str) -> None:
        """
        Sets the system prompt for the chatbot.

        Args:
            sys_prompt (str): The system prompt to set.
        """
        self.sys_prompt = sys_prompt
        
        # Si self.long_term_memory ya tiene un sys_prompt, lo actualizamos
        if len(self.long_term_memory) > 0:
            self.long_term_memory[0].content = sys_prompt
        # Si no, lo añadimos como primer elemento
        else:
            self.long_term_memory = [ConversationMessage(role="system", content=sys_prompt)]
        
        

    def add_message(self, message: ConversationMessage) -> None:
        """
        Adds a new message to the short-term memory.

        Args:
            message (ConversationMessage): The message to be added.
        """
        self.short_term_memory.add(message)
    
    def summarize(self, content: List[Union[ChatCompletionMessage, ConversationMessage]]) -> ConversationMessage:
        """
        Generates a summary of the provided conversation content using the OpenAI client.

        Args:
            content (List[Union[ChatCompletionMessage, ConversationMessage]): The conversation messages to summarize.

        Returns:
            ConversationMessage: A summarized message.
        """
        messages=[
            {
                "role": "system",
                "content": "Summarize the provided conversations using the same language."
            },
            {
                "role": "user",
                "content": f"Esta es la conversación que tienes que resumir:\n{content}\n\nSummary:"
            }
        ]
        
        out = self.llm_handler.request(messages)
        return out

    def summary_buffer(self, buffer: List[ConversationMessage], memory: CircularMemoryWithBuffer) -> ConversationMessage:
        """
        Summarizes the contents of the buffer and adds the summary to mid-term memory.

        Args:
            buffer (List[ConversationMessage]): The buffer to summarize.
            memory (CircularMemoryWithBuffer): The mid-term memory to update.

        Returns:
            ConversationMessage: The summarized message.
        """
        summarized = self.summarize(buffer)
        memory.add(summarized)
        return summarized

    def summary_lt_memory(self, buffer: List[ConversationMessage], lt_memory: List[ConversationMessage]) -> List[ConversationMessage]:
        """
        Summarizes the contents of the buffer and updates the long-term memory.

        Args:
            buffer (List[ConversationMessage]): The buffer to summarize.
            lt_memory (List[ConversationMessage]): The long-term memory to update.

        Returns:
            List[ConversationMessage]: The updated long-term memory.
        """
        summarized = self.summarize(buffer)
        if len(self.long_term_memory) == 1:
            self.long_term_memory.append(summarized)
        else:
            self.long_term_memory[1] = summarized
        return self.long_term_memory

    def get_all(self) -> List[ConversationMessage]:
        """
        Retrieves all messages from long-term, mid-term, and short-term memory in order of hierarchy.

        Returns:
            List[ConversationMessage]: All conversation messages ordered by age.
        """
        return self.long_term_memory + self.mid_term_memory.get_all() + self.short_term_memory.get_all()

# Usage
if __name__ == '__main__':
    llm_handler = ChatGPTRequester()
    msgs_memory = ChatCircularMemory(llm_handler=llm_handler)

    # client = OpenAI()

    # Get user input
    user_input = input("You: ")

    # Loop while user input is not exit nor quit
    while user_input.lower() not in ['exit', 'quit']:

        msgs_memory.add_message(ConversationMessage(role="user", content=user_input))

        # response = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=msgs_memory.get_all(),
        # )

        # msgs_memory.add_message(response.choices[0].message)
        
        response = llm_handler.request(msgs_memory.get_all())
        msgs_memory.add_message(response)

        # print(f"AI:  {response.choices[0].message.content}")
        print(f"AI:  {response.content}")

        user_input = input("You: ")
        
    tokens_prompt = llm_handler.prompt_tokens
    tokens_completion = llm_handler.completion_tokens
    tokens_total = llm_handler.total_tokens
    
    print(f"Tokens used: {tokens_total} (Prompt: {tokens_prompt}, Completion: {tokens_completion})")
