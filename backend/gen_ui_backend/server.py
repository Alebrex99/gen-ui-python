import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from gen_ui_backend.chain import create_graph
from gen_ui_backend.types import ChatInputType

# Load environment variables from .env file
load_dotenv()

#https://python.langchain.com/docs/langserve/
def start() -> None:
    app = FastAPI(
        title="Gen UI Backend",
        version="1.0",
        description="A simple api server using Langchain's Runnable interfaces",
    )

    # Configure CORS
    # back end server on localhost 8000, mentre front end su 3000
    origins = [
        "http://localhost",
        "http://localhost:3000",
    ]

    #necessario per poter accettare le richieste. vanno aggiunti tali CORS header quando chiami il server endpoint dal browser
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    #è importante definire la ROUTE per far iniziare il GRAFO e essa conterrà il nostro RUNNABLES
    # definito direttamente dentro la nostra CHAIN.PY, ovvero il GRAFO
    graph = create_graph() #creiamo il grafo cosi LANGSERVE conosce quale input e ouput ci sono; graph è un RUNNABLE

    runnable = graph.with_types(input_type=ChatInputType, output_type=dict) #è necessario, per far funzionare bene RUNNABLE,
    # di specificare i tipi in ingresso al grafo e in uscita

    #ENDPOINTS: /chat: in modo da colpirlo dal browser (grazie a FastAPI + CORS)
    #endpoints usabili dal browser:
    # /chat: per il nostro chatbot
    # POST /my_runnable/invoke - invoke the runnable on a single input
    # POST /my_runnable/batch - invoke the runnable on a batch of inputs
    # POST /my_runnable/stream - invoke on a single input and stream the output
    # POST /my_runnable/stream_log - invoke on a single input and stream the output, including output of intermediate steps as it's generated
    # POST /my_runnable/astream_events - invoke on a single input and stream events as they are generated, including from intermediate steps.
    # GET /my_runnable/input_schema - json schema for input to the runnable
    # GET /my_runnable/output_schema - json schema for output of the runnable
    # GET /my_runnable/config_schema - json schema for config of the runnable
    add_routes(app, runnable, path="/chat", playground_type="chat") #viene passato il runnable della nostra APP, qualcosa di invocabile,
    # ovvero il COMPILED GRAPH; il runnable è ciò che verrà chiamato quando colpisci l'endpoint route = /chat
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) #inizio SERVER alla porta 8000
