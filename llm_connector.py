def connect():
    print('Connecting to LLM...')

import openai
import logging
import time
import os
import pickle
import instructor
from pydantic import BaseModel
from typing import Dict, Optional

class ModelCache:
    """A class for caching model results.

    Args:
        cache_file (str): The file path to store the cache.
        cache_freq (int, optional): The frequency of persisting the cache to disk. Defaults to 5.
    """
    def __init__(self, cache_file: str, cache_freq: int = 5):
        self._cache_file = cache_file
        self._cache: Dict[str, str] = {}
        self._cache_freq = cache_freq
        self._current_cache_freq = 0

    def load(self):
        """Load the cache from the cache file."""
        try:
            with open(self._cache_file, 'rb') as file:
                self._cache = pickle.load(file)
        except FileNotFoundError:
            logging.info(f"Cache file {self._cache_file} not found. Initializing empty cache.")
            self._cache = {}

    def persist(self):
        """Persist the cache to the cache file."""
        with open(self._cache_file, 'wb') as file:
            pickle.dump(self._cache, file)

    def contains(self, key: str) -> bool:
        """Check if the cache contains a specific key.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key is in the cache, False otherwise.
        """
        return key in self._cache
    
    def get(self, key: str) -> str:
        """Get the value associated with a specific key from the cache.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            str: The value associated with the key.
        """
        return self._cache[key]
    
    def put(self, key: str, value: str) -> None:
        """Put a key-value pair into the cache.

        Args:
            key (str): The key to store the value.
            value (str): The value to be stored.
        """
        self._cache[key] = value
        self._current_cache_freq += 1
        if self._current_cache_freq >= self._cache_freq:
            self.persist()
            self._current_cache_freq = 0

class ChatGPTRequester:
    """A class for making requests to the ChatGPT model.

    Args:
        client: An instance of the OpenAI client. If None, a new client will be created using environment variables.
        model (str): The name of the ChatGPT model to use. Defaults to "gpt-4o".
        embeddings_model (str): The name of the text embeddings model to use. Defaults to "text-embedding-ada-002".
        retries (int): The number of retries to make when encountering errors. Defaults to 10.
        service_unavailable_wait_secs (float): The number of seconds to wait before retrying when encountering rate limit errors. Defaults to 0.5.
        temperature (float): The temperature parameter for generating responses. Defaults to 0.0.
        cache_file (str): The file path to store the cache. Defaults to "chatgpt_requester_cache.pkl".
        cache_freq (int): The frequency of persisting the cache to disk. Defaults to 1.
        use_cache (bool): Whether to use caching or not. Defaults to True.
    """
    
    @staticmethod
    def get_env_var(var_name: str, default: Optional[str] = None) -> Optional[str]:
        """_summary_

        Args:
            var_name (str): _description_
            default (Optional[str], optional): _description_. Defaults to None.

        Returns:
            Optional[str]: _description_
        """
        value = os.getenv(var_name, default)
        if value is None:
            logging.warning(f"Environment variable {var_name} is not set")
        return value

    def __init__(self, 
                 client=None,
                 model: str = "gpt-4o",
                 embeddings_model: str = "text-embedding-ada-002",
                 retries: int = 10,
                 service_unavailable_wait_secs: float = 0.5,
                 temperature: float = 0.0,
                 cache_file: str = "chatgpt_requester_cache.pkl",
                 cache_freq: int = 1,
                 use_cache: bool = True):
        
        if client is None:
            self._client = openai.AzureOpenAI(
                api_key=self.get_env_var("OPENAI_API_KEY"),
                api_version=self.get_env_var("OPENAI_API_VERSION"),
                azure_endpoint=self.get_env_var("AZURE_ENDPOINT"),
                azure_deployment=self.get_env_var("AZURE_DEPLOYMENT")
            )
        else:
            self._client = client

        self._model = model
        self._embeddings_model = embeddings_model
        self._retries = retries
        self._service_unavailable_wait_secs = service_unavailable_wait_secs
        self._temperature = temperature
        self._use_cache = use_cache
        self._cache = ModelCache(cache_file, cache_freq) if use_cache else None
        if self._use_cache:
            self._cache.load()
            
        # Inicializar los contadores de tokens
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        
    def reset_token_counters(self):
        """Reset the token counters to zero."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def request(self, system_message: str, user_message: str) -> str:
        """Make a request to the ChatGPT model and get a response.

        Args:
            system_message (str): The system message to provide context for the conversation.
            user_message (str): The user message to be processed.

        Raises:
            RuntimeError: If there is an error during the request.

        Returns:
            str: The response from the ChatGPT model.
        """
        result = ""
        key: str = f"{self._model}_{system_message}_{user_message}"
        if self._use_cache and self._cache.contains(key):
            result = self._cache.get(key)
        else:
            retries = self._retries
            while not result and retries > 0:
                try:
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message},
                        ], 
                        temperature=self._temperature,
                    )
                    # Actualizar los contadores de tokens
                    tokens = response.usage
                    self.prompt_tokens += tokens.prompt_tokens
                    self.completion_tokens += tokens.completion_tokens
                    self.total_tokens += tokens.total_tokens
                    
                    if response.choices[0].finish_reason == 'length':
                        raise RuntimeError("GPT truncated output")
                    result = response.choices[0].message.content
                except openai.RateLimitError:
                    logging.info("Rate limit exceeded. Waiting before retrying...")
                    # Con los valores por defecto en cada intento se duplica el tiempo de espera
                    # desde 0.5 a 256 segundos (en total, 384 segundos = 6 minutos y 24 segundos)
                    t_to_sleep = self._service_unavailable_wait_secs**((retries-self._retries)/2)
                    time.sleep(t_to_sleep)
                except openai.BadRequestError as e:
                    logging.error(f"Bad request: {e}")
                    return f"Error: Bad request - {e}"
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI error: {e}")
                    return f"Error: OpenAI error - {e}"
                except RuntimeError as e:
                    logging.error(f"Runtime error: {e}")
                    return f"Error: Runtime error - {e}"
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
                    return f"Error: Unexpected error - {e}"
                retries -= 1
        
        if result and self._use_cache:
            self._cache.put(key, result)
        return result if result else "Error: Failed to get a response after retries"
    
    def request_img_2_txt(self, system_prompt: str, data_url) -> str:
        """_summary_

        Args:
            image_path (str): _description_

        Returns:
            str: _description_
        """
        
        text=""
        retries = self._retries
        
        while not text and retries > 0:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extrae el texto de este fichero PDF"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url,
                                        "detail": "high"
                                        },
                                    },
                                ],
                            }
                        ],
                    )
            
                # Actualizar los contadores de tokens
                tokens = response.usage
                self.prompt_tokens += tokens.prompt_tokens
                self.completion_tokens += tokens.completion_tokens
                self.total_tokens += tokens.total_tokens
                        
                text = response.choices[0].message.content
            except openai.RateLimitError:
                logging.info("Rate limit exceeded. Waiting before retrying...")
                # Con los valores por defecto en cada intento se duplica el tiempo de espera
                # desde 0.5 a 256 segundos (en total, 384 segundos = 6 minutos y 24 segundos)
                t_to_sleep = self._service_unavailable_wait_secs**((retries-self._retries)/2)
                time.sleep(t_to_sleep)
            except openai.BadRequestError as e:
                logging.error(f"Bad request: {e}")
                return f"Error: Bad request - {e}"
            except openai.OpenAIError as e:
                logging.error(f"OpenAI error: {e}")
                return f"Error: OpenAI error - {e}"
            except RuntimeError as e:
                logging.error(f"Runtime error: {e}")
                return f"Error: Runtime error - {e}"
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                return f"Error: Unexpected error - {e}" 
            retries -= 1
        
        return text if text else "Error: Failed to get a response after retries"

    def request_embedding(self, text: str) -> Optional[list]:
            """Request embedding for the given text.

            Args:
                text (str): The input text to be embedded.

            Returns:
                Optional[list]: The embedding result as a list, or an error message if an error occurs.
            """
            text = text.replace("\n", " ")
            retries = self._retries
            embedding_result = None
            while retries > 0:
                try:
                    embedding_result = self._client.embeddings.create(input=[text], model=self._embeddings_model)
                    
                    # Actualizar los contadores de tokens
                    tokens = embedding_result.usage
                    self.prompt_tokens += tokens.prompt_tokens
                    self.completion_tokens += tokens.completion_tokens
                    self.total_tokens += tokens.total_tokens
                
                    return embedding_result.data[0].embedding
                except openai.RateLimitError:
                    logging.info("Rate limit exceeded. Waiting before retrying...")
                    time.sleep(self._service_unavailable_wait_secs)
                except openai.BadRequestError as e:
                    logging.error(f"Bad request: {e}")
                    return f"Error: Bad request - {e}"
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI error: {e}")
                    return f"Error: OpenAI error - {e}"
                except RuntimeError as e:
                    logging.error(f"Runtime error: {e}")
                    return f"Error: Runtime error - {e}"
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
                    return f"Error: Unexpected error - {e}"
                retries -= 1
            return embedding_result if embedding_result else "Error: Failed to get embedding after retries"
    


    def request_ner(self, system_message: str, user_message: str, output_structure: BaseModel) -> str:
        """
        Extracts structured data from natural language using OpenAI chat completions.

        Args:
            system_message (str): The system message to provide context for the conversation.
            user_message (str): The user message to be processed.
            output_structure (BaseModel): The structure of the output data.

        Returns:
            str: The extracted structured data as a string.
        """
        
        # Patch the OpenAI client
        
        client = instructor.from_openai(self._client)

        # Extract structured data from natural language
        # user_info = client.chat.completions.create(
        user_info, completion = client.chat.completions.create_with_completion(
            model=self._model,
            response_model=output_structure,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                ])
        
        # Actualizar los contadores de tokens
        tokens = completion.usage
        self.prompt_tokens += tokens.prompt_tokens
        self.completion_tokens += tokens.completion_tokens
        self.total_tokens += tokens.total_tokens
        
        return user_info #, completion
