import logging
import os
import time

import dotenv

from .base_language_model import BaseLanguageModel

import google.generativeai as genai 

#from google import genai

logger = logging.getLogger(__name__)
# Disable OpenAI and httpx logging
# Configure logging level for specific loggers by name
#ITC_MODIFICADO (todo el script es nuevo)

dotenv.load_dotenv()


class Gemini(BaseLanguageModel):
    """A class that interacts with OpenAI's ChatGPT models through their API.

    This class provides functionality to generate text using ChatGPT models while handling
    token limits, retries, and various input formats.

    Args:
        model_name_or_path (str): The name or path of the ChatGPT model to use
        retry (int, optional): Number of retries for failed API calls. Defaults to 5

    Attributes:
        retry (int): Maximum number of retry attempts for failed API calls
        model_name (str): Name of the ChatGPT model being used
        maximun_token (int): Maximum token limit for the specified model
        client (OpenAI): OpenAI client instance for API interactions

    Methods:
        token_len(text): Calculate the number of tokens in a given text
        generate_sentence(llm_input, system_input): Generate response using the ChatGPT model

    Raises:
        KeyError: If the specified model is not found when calculating tokens
        Exception: If generation fails after maximum retries
    """

    def __init__(self, model_name_or_path: str, retry: int = 5):
        self.retry = retry
        self.model_name = model_name_or_path
        self.maximun_token: int = 12288

        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.client = genai.GenerativeModel(self.model_name)
        

    def token_len(self, text: str) -> int:
        """Returns the number of tokens used by a list of messages."""
        
        return self.client.count_tokens(text).total_tokens

    def generate_sentence(
        self, llm_input: str | list, system_input: str = ""
    ) -> str | Exception:
        """Generate a response using the ChatGPT API.

        This method sends a request to the ChatGPT API and returns the generated response.
        It handles both single string inputs and message lists, with retry logic for failed attempts.

        Args:
            llm_input (Union[str, list]): Either a string containing the user's input or a list of message dictionaries
                in the format [{"role": "role_type", "content": "message_content"}, ...]
            system_input (str, optional): System message to be prepended to the conversation. Defaults to "".

        Returns:
            Union[str, Exception]: The generated response text if successful, or the Exception if all retries fail.
                The response is stripped of leading/trailing whitespace.

        Raises:
            Exception: If all retry attempts fail, returns the last encountered exception.

        Notes:
            - Automatically truncates inputs that exceed the maximum token limit
            - Uses exponential backoff with 30 second delays between retries
            - Sets temperature to 0.0 for deterministic outputs
            - Timeout is set to 60 seconds per API call
        """

        # If the input is a list, it is assumed that the input is a list of messages
        if isinstance(llm_input, list):
            message = llm_input
        else:
            message = []
            if system_input:
                message.append({"role": "system", "content": system_input})
            message.append({"role": "user", "content": llm_input})
        cur_retry = 0
        num_retry = self.retry
        # Check if the input is too long
        message_string = "\n".join([m["content"] for m in message])
        input_length = self.token_len(message_string)
        if input_length > self.maximun_token:
            print(
                f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens."
            )
            llm_input = llm_input[: self.maximun_token]
        error = Exception("Failed to generate sentence")
        while cur_retry <= num_retry:
            try:
                response = self.client.generate_content(
                contents=[
                            {
                                "role": "user",  # El rol de qui està enviant el missatge
                                "parts": [
                                            {"text": message_string}  # El contingut del missatge
                                        ]
                            }
                        ],
                generation_config={"temperature": 0.0}  # Opció de temperatura
                )

                result = response.text.strip()  # type: ignore
                return result
            except Exception as e:
                logger.error("Message: ", llm_input)
                logger.error("Number of token: %s", self.token_len(message_string))
                logger.error(e)
                time.sleep(30)
                cur_retry += 1
                error = e
                continue
        return error
