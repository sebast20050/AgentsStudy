# RAG_DEMO

Este proyecto tiene por objetivo poder demostrar la evolucion de agentes conversacionales sobre datos estructurados. Es decir, en vez de generar un dashboard BI tipico para poder obtener conocimiento sobre un dataset, entonces, lo que se hace se genera una interfaz conversacional, a fin de demostrar que la adquisicion de conocimiento puede ser muy efectiva con el uso de estos agentes y disponibilizados por un canal de comunicacion de uso cotidiano como es Whatsapp en nuestra region. 


### Estructura General de la Solucion

La solucion software se basa en un unico agente conversacional, el cual evoluciona en su complejidad a lo largo de los diferentes modelos utilizados. 
Genere un archivo de funciones `Rag_funcions.py` que contiene un conjunto de funciones que se utilizan en todos los archivos. 
Para los datos, he utilizado la tabla publica del arbolado lineal de la Ciudad de Buenos Aires que se obtiene desde: https://buenosaires.gob.ar/espaciopublicoehigieneurbana/gestion-comunal/arbopedia/arbolado-de-la-ciudad-0 que se encuentra en `arbolado-publico-lineal-2017-2018.csv` 

Posteriormente he generado un archivo csv: `definicion.csv` sobre el cual defino las columnas/variables de la fuentes de datos, y su significado. En caso de que la variable no tenga explicación, ese campo no se importa, porque se considera que no contriubuye al conocimiento del agente. 

Para el agente utilizo una solucion del tipo fastapi, esta solucion debe contener siempre los webhooks: `startup`, `get` y `post`.
Para la ejecucion del agente, el comando que utliizo: `uvicorn OpenAI:app --host 0.0.0.0 --port 8000 --reload` que genera el servidor web en el host, y luego expongo el puerto con ngrok: `ngrok http 8000`

Para whatsapp utilizo la API de Meta, que obtengo directamente desde https://developers.facebook.com/apps alli, configuro el webhook con la direccion que expone ngrok + `/webhook`. Obviamente alli registro los numeros que se contactaron con el numero de prueba para hacer las interacciones. 

Para el LLM estoy usando la version gratis del modelo `gemini-2.5-flash`, sobre el cual lo unico que se requiere es ingresar a: https://aistudio.google.com/projects y de alli generar un Token que se incluye en el .env

### Langchain.py

Este código habilita un agente LangChain experimental que recibe directamente un Pandas Dataframe. Lo que hace Langchain basicamente es definir un contexto al LLM que contiene las columnas del dataframe, los tipos de datos, y las primeras filas, para que el LLM genere el codigo python de la consulta, y devuelva la informacion al usuario. Esto es mas performante que pedir al LLM que genere codigo SQL y que consulte el Dataframe. Brinda muy buenos resultados, para un primer POC. 

### OpenAI.py

Este código muy similar al anterior, pero en vez de utilizar LangChain utilizo un Agente directo de OpenAI. Como estoy usando la versión gratis de Gemini 🐭, entonces tengo que hacerle un sampleo antes porque por la cantidad de registros que tiene el dawtaset completo, excedo la kuota. 
Este en general es bastante mejor que el de LangChain, porque puede reconocer mejor el flujo de la conversación y no hace tantas alucinaciones. 


### FAISS

Este codigo lo que hace es crear un indice de vectores de embeddings por medio de FAISS. Esta generación toma su tiempo, por lo que tuve que guardarlo en un archivo .bin junto con su corpus en .txt, ahora bien, como se puede observar en su ejecución, el mismo no responde con el accuracy de los anteriores. Y esto es porque en la generación de los embeddings, le paso todo sin hacer un analisis previo de nada. Esto es bien importante, porque entonces aqui el contexto tiene que tener una gestion previa de inteligencia que los anteriores ya de alguna forma asumian. En este caso no sucede. Con lo cual se requiere realizar un proceso previo de datascience a fin de que el agente responda adecuadamente. 

### Human in the Loop. 

El objetivo de este aprendizaje es el de incluir un humano dentro de la conversacion. Para ello he construido un frontend simple que pueda interactuar con el backend de forma que permita esta colaboracion en la conversacion. 
