import json
from typing import List, Tuple, Dict, Any
import openai
import os
import sys

# OBSOLETO: Este archivo ha sido reemplazado por gaiatec/scripts/examen.py
# Usar: cd gfm-rag-main && uv run python ../gaiatec/scripts/examen.py --preguntas preguntas.txt
#
#-----------------------------------------------------------IMPORTANTE----------------------------------------------------------------------------#
#HAY UNA VARIABLE IMPORTANTE A TENER EN CUENTA Y ES EL RANKER QUE SE EMPLEA
#PARA PUNTUAR Y, POR ENDE, ESCOGER Y DEVOLVER LOS DOCUMENTOS MÁS RELEVANTES

#ESTE PARÁMETRO SE PUEDE CONFIGURAR EN gfmrag/workflow/config/stage3_qa_ircot_inference.yaml

#AQUÍ ESTÁN LOS YAMLS DE CONFIGURACIÓN DE TODOS LOS TIPOS DE RANKERS DISPONIBLES: gfmrag/workflow/config/doc_ranker/
#AQUÍ LA IMPLEMENTACIÓN DE CADA UNO: gfmrag/doc_rankers.py
#EMPLEAMOS 2 TIPOS DE RANKERS (SON FUNCIONES DETERMINISTAS y NO MODELOS DE IA):

#idf_topk_ranker
#topk_ranker 

#CADA UNO TIENE SUS PROPIEDADES Y, POR TANTO, ÁMBITOS DE USO DISTINTO
#----------------------------------------------------------------------------------------------------------------------------------------------------#
#RUTA DEL ARCHIVO TXT CON LAS PREGUNTAS DEL EXAMEN
PREGUNTAS_EXAMEN = "/home/compartit/gaiatec/documentaciónRagGaiatec/materialGFM/trainAndTest/examen22012026"

#APARTADOS A RESPONDER POR CADA PREGUNTA
CAMPOS_PREGUNTA = [
    "Pregunta", #string
    "Respuesta", #string
    "question_entities", #lista de strings
    "Nº_Documentos", #int
    "Documentos_recuperados", #lista de strings        
]

#RUTA DEL RESULTADO DE EVALUACIÓN
REPUESTAS_EXAMEN = "/home/compartit/gaiatec/documentaciónRagGaiatec/materialGFM/trainAndTest/examen_2.1PrensadoOrdinaria2/TopKRanker/respuesta22012026"

#ALUMNO GAIATEC (LLM QUE RECIBE LA PREGUNTA Y LOS DOCUMENTOS RECUPERADOS DE GFM)
#CONSIDERACIÓN: SE CREA UN ÚNICO ALUMNO, NO DEBEMOS CREAR UNO POR CADA PREGUNTA

#---------------------------------------------------------------------------------------------------
if "OPENAI_API_KEY" not in os.environ:
    os.environ['OPENAI_API_KEY'] = 'sk-DUMMYKEYFORLOCALSERVER'


GAIATEC = openai.OpenAI(
        base_url="http://172.16.128.76:8082/v1",  # Solo hasta '/v1'
        api_key="sk-no-key-required"  # Algunas implementaciones compatibles requieren una clave no vacía
    )
#------------------------------------------------------------------------------------------------------

#HOJA DE EXAMEN
lista_de_diccionarios = []


#CONFIGURACIÓN DEL GFM Y ARREGLOS
#-----------------------------------------------------------------------------------------------------------------------------
def setup_environment():
    nvjitlink_path = "/home/compartit/gaiatec/GFM-prova/gfm-rag-main/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib"
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    
    if nvjitlink_path not in current_ld_path:
        # Configurar el entorno y reiniciar el script
        if current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{nvjitlink_path}:{current_ld_path}"
        else:
            os.environ['LD_LIBRARY_PATH'] = nvjitlink_path
        
        # Reiniciar el script con el nuevo entorno
        os.execv(sys.executable, [sys.executable] + sys.argv)

setup_environment()
#--------------------------------------------------------------------------------------------------------------------------------
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from hydra.utils import instantiate
from gfmrag.llms import BaseLanguageModel
from gfmrag.prompt_builder import QAPromptBuilder
from gfmrag import GFMRetriever
#----------------------------------------------------------------------------------------------------------------------------------
GPU = 1
TOP_K = 8
CONFIG_PATH = "gfmrag/workflow/config"
CONFIG_NAME = "stage3_qa_ircot_inference"
initialize(CONFIG_PATH, version_base = "1.1")
torch.cuda.set_device(GPU)

#SE CREA EL GFMRAG_RETRIEVER
GFM = GFMRetriever.from_config(compose(CONFIG_NAME))
#------------------------------------------------------------------------------------------------------------------------------------

def contestaUnaPregunta(query: str) -> tuple[str, list, list, str]:

    docs, enti = GFM.retrieve(query, TOP_K)

    try:
        response = GAIATEC.chat.completions.create(
        model="GaiatecMistralSmall",
        messages=[
            {"role": "system", "content": "Eres un experto profesor en tecnología cerámica. Responde a la pregunta en función de los documentos aportados"},
            {"role": "user", "content": query + str(docs)}
        ]
         )

        print("Respuesta")
        respuesta_llm = response.choices[0].message.content
        return query, respuesta_llm, enti, TOP_K, [dic["title"] for dic in docs]

    except Exception as e:
        print(f"Ocurrió un error: {e}")


def main():

    try:
        with open(PREGUNTAS_EXAMEN, 'r', encoding='utf-8') as file:
            for linea in file:

                pregunta = linea.strip()
                print(pregunta)

                if pregunta:

                    resultado_tupla = contestaUnaPregunta(pregunta)
                    nuevo_diccionario = dict(zip(CAMPOS_PREGUNTA, resultado_tupla))
                    lista_de_diccionarios.append(nuevo_diccionario)

        with open(REPUESTAS_EXAMEN, 'w', encoding='utf-8') as json_file:
            json.dump(lista_de_diccionarios, json_file, indent=4, ensure_ascii=False)


    except FileNotFoundError:
        print(f"ERROR: No se pudo encontrar el archivo {PREGUNTAS_EXAMEN}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")



if __name__ == "__main__": 
    main()