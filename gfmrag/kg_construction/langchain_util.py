import os
from typing import Any

from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

from langchain_google_genai import ChatGoogleGenerativeAI #ITC_MODIFICADO

def init_langchain_model(
    llm: str,
    model_name: str,
    temperature: float = 0.0,
    max_retries: int = 5,
    timeout: int = 60,
    **kwargs: Any,
) -> ChatOpenAI | ChatTogether | ChatOllama | ChatLlamaCpp | ChatGoogleGenerativeAI:
    """
    Initialize a language model from the langchain library.
    :param llm: The LLM to use, e.g., 'openai', 'together'
    :param model_name: The model name to use, e.g., 'gpt-3.5-turbo'
    """
    if llm == "openai":
        # https://python.langchain.com/v0.1/docs/integrations/chat/openai/
        
        
        # ELIMINAR O COMENTAR la siguiente aserción ya que GaiatecMistralSmall no sigue este patrón:
        # assert model_name.startswith("gpt-")
        
        # Se asume que 'base_url' vendrá dentro de los **kwargs desde la configuración YAML
        # Si 'base_url' no está en kwargs, usará el valor por defecto (API oficial de OpenAI)



        # Obtener base_url (si existe, es el servidor local)
        base_url = kwargs.pop("base_url", None)
        
        # Determinar la clave API:
        if base_url:
            # Si es servidor local, no pasamos clave (o pasamos None/cadena vacía) 
            # para evitar que la rechace. La cadena vacía ('') es el mejor intento.
            api_key_to_use = "" 
        else:
            # Si es la API oficial, sí necesitamos la clave.
            api_key_to_use = os.environ.get("OPENAI_API_KEY")


        return ChatOpenAI(
            api_key=api_key_to_use,
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            **kwargs,
        )
    elif llm == "nvidia":
        # https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/

        return ChatNVIDIA(
            nvidia_api_key=os.environ.get("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
            model=model_name,
            temperature=temperature,
            **kwargs,
        )
    elif llm == "together":
        # https://python.langchain.com/v0.1/docs/integrations/chat/together/

        return ChatTogether(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            model=model_name,
            temperature=temperature,
            **kwargs,
        )
    elif llm == "ollama":
        # https://python.langchain.com/v0.1/docs/integrations/chat/ollama/

        return ChatOllama(model=model_name)  # e.g., 'llama3'
    elif llm == "llama.cpp":
        # https://python.langchain.com/v0.2/docs/integrations/chat/llamacpp/

        return ChatLlamaCpp(
            model_path=model_name, verbose=True
        )  # model_name is the model path (gguf file)
    elif llm == "google":
        # ITC_MODIFICADO
        return ChatGoogleGenerativeAI(model = model_name, temperature=temperature, api_key=os.environ.get("GOOGLE_API_KEY"),
            **kwargs)

    else:
        # add any LLMs you want to use here using LangChain
        raise NotImplementedError(f"LLM '{llm}' not implemented yet.")