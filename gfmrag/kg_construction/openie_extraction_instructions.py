from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

## General Prompts
one_shot_passage = """Radio City es la primera emisora de radio FM privada de la India y fue inaugurada el 3 de julio de 2001.
Reproduce canciones en hindi, inglés y lenguas regionales.
Radio City se adentró recientemente en los Nuevos Medios en mayo de 2008 con el lanzamiento de un portal musical —PlanetRadiocity.com— que ofrece noticias relacionadas con la música, vídeos, canciones y otras funciones musicales.
"""

one_shot_passage_entities = """{
  "entidades_nombradas": [
    "Radio City",
    "India",
    "3 de julio de 2001",
    "Hindi",
    "Inglés",
    "mayo de 2008",
    "PlanetRadiocity.com"
  ]
}

"""

## NER Prompts

ner_instruction = """Extrae únicamente las entidades nombradas de tipo científico del siguiente párrafo.
Responde con una lista en formato JSON.
Sigue estrictamente el formato JSON requerido.
"""

ner_input_one_shot = f"""Paragraph:
```
{one_shot_passage}
```
"""

ner_output_one_shot = one_shot_passage_entities

ner_user_input = "Paragraph:```\n{user_input}\n```"
ner_prompts = ChatPromptTemplate.from_messages(
    [
        SystemMessage(ner_instruction),
        HumanMessage(ner_input_one_shot),
        AIMessage(ner_output_one_shot),
        HumanMessagePromptTemplate.from_template(ner_user_input),
    ]
)

## Post NER OpenIE Prompts

one_shot_passage_triples = """{
  "triples": [
    ["Radio City", "ubicada en", "India"],
    ["Radio City", "es", "emisora de radio FM privada"],
    ["Radio City", "empezó el", "3 de julio de 2001"],
    ["Radio City", "reproduce canciones en", "Hindi"],
    ["Radio City", "reproduce canciones en", "Inglés"],
    ["Radio City", "se adentró en", "Nuevos Medios"],
    ["Radio City", "lanzó", "PlanetRadiocity.com"],
    ["PlanetRadiocity.com", "lanzado en", "mayo de 2008"],
    ["PlanetRadiocity.com", "es", "portal musical"],
    ["PlanetRadiocity.com", "ofrece", "noticias"],
    ["PlanetRadiocity.com", "ofrece", "vídeos"],
    ["PlanetRadiocity.com", "ofrece", "canciones"]
  ]
}

"""

openie_post_ner_instruction = """Construye un grafo RDF (Marco de Descripción de Recursos) a partir de los pasajes y las listas de entidades nombradas proporcionadas.
Responde con una lista JSON de tripletes, donde cada tripleta representa una relación en el grafo RDF.

Presta atención a los siguientes requisitos:
-Cada tripleta debe contener al menos una, pero preferiblemente dos, de las entidades nombradas en la lista correspondiente a cada pasaje.
-Resuelve claramente los pronombres con sus nombres específicos para mantener la claridad.

"""

openie_post_ner_frame = """Convierte el párrafo en un diccionario JSON.
Debe contener una lista de entidades nombradas y una lista de tripletas.
Párrafo:


```
{passage}
```

{named_entity_json}
"""

openie_post_ner_input_one_shot = openie_post_ner_frame.replace(
    "{passage}", one_shot_passage
).replace("{named_entity_json}", one_shot_passage_entities)

openie_post_ner_output_one_shot = one_shot_passage_triples

openie_post_ner_prompts = ChatPromptTemplate.from_messages(
    [
        SystemMessage(openie_post_ner_instruction),
        HumanMessage(openie_post_ner_input_one_shot),
        AIMessage(openie_post_ner_output_one_shot),
        HumanMessagePromptTemplate.from_template(openie_post_ner_frame),
    ]
)
