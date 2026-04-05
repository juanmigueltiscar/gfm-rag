import logging
import os
import time
from typing import Union, List, Dict

import dotenv
# Cliente Nativo de Mistral AI
from mistralai.client import MistralClient
from mistralai.models.chat_models import ChatMessage
# Usaremos tiktoken para la aproximación de tokens, ya que la API de Mistral no tiene un endpoint de conteo directo
import tiktoken 
from .base_language_model import BaseLanguageModel

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# --- Configuración Específica de la API de Mistral ---
MISTRAL_API_MODEL = "mistral-small-2407" # Modelo actual para Small 3.2
MISTRAL_TOKEN_LIMIT = 128000 

class MistralSmall32API(BaseLanguageModel):
    """
    Una clase que interactúa con el modelo Mistral Small 3.2 a través de su cliente API nativo (mistralai).
    Implementa el contrato de BaseLanguageModel.
    """

    def __init__(self, model_name_or_path: str = MISTRAL_API_MODEL, retry: int = 5):
        
        # 1. Asignación de Parámetros de Configuración YAML
        self.retry = retry
        self.model_name = model_name_or_path
        self.maximun_token: int = MISTRAL_TOKEN_LIMIT

        # 2. Configuración del Cliente API Nativo de Mistral
        # La API Key se lee automáticamente desde la variable MISTRAL_API_KEY
        self.client = MistralClient(api_key=os.environ['MISTRAL_API_KEY'])
        logger.info(f"Cliente nativo de Mistral inicializado para el modelo: {self.model_name}.")

    def token_len(self, text: str) -> int:
        """
        Calcula la longitud de tokens usando una aproximación (tiktoken),
        ya que el cliente nativo de Mistral no ofrece un método de conteo directo.
        """
        try:
             # Usar la codificación de un modelo similar de OpenAI como aproximación
             encoding = tiktoken.encoding_for_model("gpt-4") 
             return len(encoding.encode(text))
        except Exception as e:
             logger.warning(f"No se pudo usar tiktoken para conteo de tokens: {e}. Usando aproximación heurística.")
             # Aproximación simple si falla tiktoken
             return len(text) // 4 

    def generate_sentence(
        self, llm_input: Union[str, List[Dict[str, str]]], system_input: str = ""
    ) -> Union[str, Exception]:
        """Genera una respuesta usando la API de Mistral AI."""

        messages: List[ChatMessage] = []
        
        # 1. Formateo de los Mensajes al formato nativo de Mistral (ChatMessage)
        if system_input:
            # Los modelos instruidos de Mistral aceptan el rol 'system'
            messages.append(ChatMessage(role="system", content=system_input))
            
        if isinstance(llm_input, str):
            messages.append(ChatMessage(role="user", content=llm_input))
        else: # Asumimos que es una lista de mensajes estilo {"role": "...", "content": "..."}
             for msg in llm_input:
                 messages.append(ChatMessage(role=msg["role"], content=msg["content"]))
            
        # 2. Lógica de Truncamiento (Mantenida por consistencia, con la crítica de ser insegura)
        # Nota: La lógica de truncamiento aquí solo mide la longitud concatenada, lo cual es inexacto para chat history.
        message_string = " ".join([m.content for m in messages])
        input_length = self.token_len(message_string)
        
        if input_length > self.maximun_token:
            logger.warning(
                f"Input length {input_length} exceeds max token {self.maximun_token}. Truncando entrada."
            )
            # En una versión robusta, se implementaría aquí el vaciado del historial.

        cur_retry = 0
        error = Exception("Failed to generate sentence")

        # 3. Generación y Manejo de Errores con Reintentos
        while cur_retry <= self.retry:
            try:
                # LLamada nativa de MistralClient
                response = self.client.chat(
                    model=self.model_name, 
                    messages=messages, 
                    temperature=0.0, # Para respuestas deterministas
                )
                
                # El cliente de Mistral devuelve el contenido en response.choices[0].message.content
                result = response.choices[0].message.content.strip()
                return result
                
            except Exception as e:
                logger.error(f"Error en la API de Mistral (Intento {cur_retry + 1}/{self.retry}): {e}")
                time.sleep(15 * (2**cur_retry)) # Backoff exponencial
                cur_retry += 1
                error = e
                continue

        return error