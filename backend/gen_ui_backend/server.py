import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from gen_ui_backend.chain import create_graph
from gen_ui_backend.types import ChatInputType

# Load environment variables from .env file
load_dotenv()

# Create the FastAPI app outside the function for Vercel deployment
app = FastAPI(
    title="Gen UI Backend",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
# Aggiungi l'URL del tuo frontend su Vercel
"https://your-frontend-domain.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the graph and add routes
graph = create_graph()
runnable = graph.with_types(input_type=ChatInputType, output_type=dict)
add_routes(app, runnable, path="/chat", playground_type="chat")

# For local development
def start() -> None:
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# For Vercel deployment
# In Vercel, the app object will be imported directly
if __name__ == "__main__":
    start()
