from llm_execenvtool import LLMExecEnvTool
from memory import ChatMemory
from llm_circular_memory import ChatCircularMemory
from llm_handler import ChatGPTRequester

# msgs_memory = ChatMemory(word_limit=20000, n_recent=10)
llm_handler = ChatGPTRequester()
msgs_memory = ChatCircularMemory(llm_handler=llm_handler)

llm_exec = LLMExecEnvTool(llm_handler = llm_handler, memory=msgs_memory, imports = ['joke_cat_dog'])

def main():
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
    main()