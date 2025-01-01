from llm_execenvtool import LLMExecEnvTool
from memory import ChatMemory

msgs_memory = ChatMemory(word_limit=20000, n_recent=10)

llm_exec = LLMExecEnvTool(memory=msgs_memory, imports = ['joke_cat_dog'])

def main():
    # Get user input
    user_input = input("You: ")
    
    # Loop while user input is not exit nor quit 
    while user_input.lower() not in ['exit', 'quit']:
        
        response = llm_exec.answer_user_request(user_input)
        
        print(response)
        
        user_input = input("You: ")

if __name__ == '__main__':
    main()