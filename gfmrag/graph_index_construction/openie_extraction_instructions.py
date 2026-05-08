from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

## General Prompts
one_shot_passage = """Impresión inkjet en cerámica
La impresión inkjet es una tecnología de decoración cerámica ampliamente adoptada en España.
El sistema piezoeléctrico controla la eyección de gotas de tinta sobre la superficie de la baldosa.
Las empresas de Castellón fueron pioneras en su adopción a partir del año 2003.
El cabezal de impresión es el componente clave de la máquina industrial."""

one_shot_passage_entities = """{"named_entities":
    ["impresión inkjet", "decoración cerámica", "España", "sistema piezoeléctrico", "gotas de tinta", "baldosa", "empresas de Castellón", "2003", "cabezal de impresión", "máquina industrial"]
}
"""

## NER Prompts

ner_instruction = """Tu tarea es extraer entidades nombradas del párrafo dado.
Responde con una lista JSON de entidades.
Sigue estrictamente el formato JSON requerido.
"""

ner_input_one_shot = f"""Párrafo:
```
{one_shot_passage}
```
"""

ner_output_one_shot = one_shot_passage_entities

ner_user_input = "Párrafo:```\n{user_input}\n```"
ner_prompts = ChatPromptTemplate.from_messages(
    [
        SystemMessage(ner_instruction),
        HumanMessage(ner_input_one_shot),
        AIMessage(ner_output_one_shot),
        HumanMessagePromptTemplate.from_template(ner_user_input),
    ]
)

## Post NER OpenIE Prompts

one_shot_passage_triples = """{"triples": [
            ["impresión inkjet", "es una tecnología de", "decoración cerámica"],
            ["impresión inkjet", "fue adoptada en", "España"],
            ["sistema piezoeléctrico", "controla la eyección de", "gotas de tinta"],
            ["gotas de tinta", "se depositan sobre", "baldosa"],
            ["empresas de Castellón", "fueron pioneras en adopción de", "impresión inkjet"],
            ["empresas de Castellón", "adoptaron la tecnología desde", "2003"],
            ["cabezal de impresión", "es el componente clave de", "máquina industrial"]
    ]
}
"""

openie_post_ner_instruction = """Tu tarea es construir un grafo RDF (Resource Description Framework) a partir de los pasajes dados y las listas de entidades nombradas.
Responde con una lista JSON de tripletas, donde cada tripleta representa una relación en el grafo RDF.

Presta atención a los siguientes requisitos:
- Cada tripleta debe contener al menos una, pero preferiblemente dos, de las entidades nombradas de la lista para cada pasaje.
- Resuelve claramente los pronombres a sus nombres específicos para mantener la claridad.

"""

openie_post_ner_frame = """Convierte el párrafo en un diccionario JSON con una lista de entidades nombradas y una lista de tripletas.
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
