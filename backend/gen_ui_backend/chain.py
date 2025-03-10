from typing import List, Optional, TypedDict

from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

from gen_ui_backend.tools.github import github_repo
from gen_ui_backend.tools.invoice import invoice_parser
from gen_ui_backend.tools.weather import weather_data

"""Questo modulo contiene la catena di esecuzione per la generazione di grafici. cioè contiene
il LANG GRAPH chain, quindi la catena sequenziale ad alto livello"""

#Tale classe rappresenta lo STATO passato tra i nodi del langGraph =
# sottoforma di dizionario di TypedDict (ogni chiave è un attr della classe)
#contiene il messaggio dell'utente, il risultato dell'elaborazione e le chiamate ai tool:
# {"input": HumanMessage , "result": str or None , "tool_calls": [ … ] or None , "tool_result": { … } or None}
class GenerativeUIState(TypedDict, total=False):
    input: HumanMessage #input dell'utente
    result: Optional[str] #è una stringa se LLM non chiama un tool
    """Plain text response if no tool was used."""
    tool_calls: Optional[List[dict]] #è una lista di dizionari (oggetti tool) se LLM chiama un tool/ vari tools altrimenti None se ritorna il testo
    """A list of parsed tool calls."""
    tool_result: Optional[dict] #se LLm chiama un tool, allora verrà chiamata la invoke tools e verrà restituito il risultato della funzione del tool (un dizionario)
    # usato poi per aggiornare la chat history. (puoi provare a modificare il result con una lista di dizionari, in modo da poter gestire più tool calls)
    """The result of a tool call."""


def invoke_model(state: GenerativeUIState, config: RunnableConfig) -> GenerativeUIState:
    tools_parser = JsonOutputToolsParser() #OpenAI tools output parser, il messaggio dell'AI sarà in JSON
    sysmsg = SystemMessage("You are a helpful assistant. You're provided a list of tools, and an input from the user.\n") #puoi sostituire a 'system' questo nel prompt
    initial_prompt = ChatPromptTemplate.from_messages( #creare il prompt da inviare al modello con le istruzioni (system e input)
        [
            (
                "system",
                "You are a helpful assistant. You're provided a list of tools, and an input from the user.\n"
                + "Your job is to determine whether or not you have a tool which can handle the users input, or respond with plain text.",
            ),
            MessagesPlaceholder("input"), #qui è dove l'Input utente della chat history deve finire,
        ]
    )
    print(state["input"]) #stampa l'input dell'utente
    model = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
    tools = [github_repo, invoice_parser, weather_data] #import dei tools, decorati con @tool, delle funzioni definite nei pythons modules
    model_with_tools = model.bind_tools(tools) #interfaccia per connettere i tools al modello (chiamato sul modello stesso)
    chain = initial_prompt | model_with_tools #uso di LCEL (lang chain expression language) vengono unificati due RUNNABLES in uno solo (eseguiti con invoke in sequenza uno dopo l'altro, usando output come input del successivo)
    result = chain.invoke({"input": state["input"]}, config) #n.b. accediamo a input con state["input"], perchè state è istanza che eredita da typedDict
    #tale INVOKE avvia la catena di decisione partendo dall'input. il result conterrà la risposta del LLM, quindi o le tool_calls
    # (attributo dell'obj, cioè lista di calls con gli args da usare per chiamare il tool), oppure il plain text response
    print(result) #stampa il risultato del modello
    #-------------PARSING LOGIC: formattatore output----------------
    #la risposta del chat model è un AIMessage, quindi se il risultato non è AIMessage, solleva un'eccezione
    if not isinstance(result, AIMessage):
        raise ValueError("Invalid result from model. Expected AIMessage.")
    if isinstance(result.tool_calls, list) and len(result.tool_calls) > 0: #se il tool_calls dl result è una List (lista)
        parsed_tools = tools_parser.invoke(result, config) #il risultato viene elaborato con un JSON parser,
        # viene usato per popolare il campo dello stato "tool_calls"; se non ci sono tool calls, allora il risultato è una stringa (campo result)
        print(parsed_tools) #stampa il risultato del parser
        return {"tool_calls": parsed_tools} #viene restituito un nuovo stato, ciò presente prima è perso; se vuoi fare update dovresti fare: return {**state, "tool_calls": parsed_tools}
    else:
        return {"result": str(result.content)}


def invoke_tools_or_return(state: GenerativeUIState) -> str:
    if "result" in state and isinstance(state["result"], str): #se il risultato è una stringa, allora ritorna END (elemento langgraph per non chiamare altri nodi)
        return END
    elif "tool_calls" in state and isinstance(state["tool_calls"], list): #in tal caso ritorna il nodo "invoke_tools" che è invocato perchè tornato come stringa nell'EDGE
        return "invoke_tools"
    else:
        raise ValueError("Invalid state. No result or tool calls found.")


def invoke_tools(state: GenerativeUIState) -> GenerativeUIState:
    #mapping funzioni con nomi per poi invocarli. verra usato lo STATE per invocare la funzione corretta
    tools_map = {
        "github-repo": github_repo,
        "invoice-parser": invoice_parser,
        "weather-data": weather_data,
    }

    if state["tool_calls"] is not None:
        print(f"state[tool_calls]: {state['tool_calls']}") #LISTA: [{"name": "weather-data", "args": {"city": "Rome", "state": "NY", "country": "usa"}, "type": "weather-data"}, {"name": "weather-data", "args": {"city": "Rome", "state": "NY", "country": "usa"}, "type": "weather-data"}]
        #POSSIBILITA: MANAGE N TOOL_CALLS
        #tools = state["tool_calls"]
        #tool_result=None
        #for t in tools:
        #    tool = t["type"]
        #    selected_tool = tools_map[tool]
        #    tool_result = selected_tool.invoke(t["args"])
        #print(tool_result) #stampa l'ultimo tool. la renderizzazione UI forse si sovrappone
        #return {"tool_result": tool_result}
        #PARTE CORRETTA:
        tool = state["tool_calls"][0] #QUI VIENE PRESO SOLO IL PRIMO TOOL CALL, PUOI MODIFICARE PER GESTIRE PIU TOOL CALLS
        selected_tool = tools_map[tool["type"]] #type associato al name del tool
        print(f"args: {tool['args']}") #stampa gli args del tool
        return {"tool_result": selected_tool.invoke(tool["args"])} #viene eseguita e ritornato, con invoke, l'output di esecuzione della funzione del tool, da fare sempre con tool.invoke(args)
    else:
        raise ValueError("No tool calls found in state.")

#funzione di creazione del grafo
def create_graph() -> CompiledGraph:
    workflow = StateGraph(GenerativeUIState) #tipologia di grafo statico (non controllato dall'LLM)
    # ci sono vari nodi, corrispondono ai blocchi generali dello schema principale
    workflow.add_node("invoke_model", invoke_model)  # type: ignore
    workflow.add_node("invoke_tools", invoke_tools)
    workflow.add_conditional_edges("invoke_model", invoke_tools_or_return) #crea un arco condizionale tra i nodi, in base alla funzione passata. inizia da invoke_model e conduce al nodo passato dalla funzione passata
    workflow.set_entry_point("invoke_model")
    workflow.set_finish_point("invoke_tools")

    graph = workflow.compile() #tale grafo viene compilato e restituito al langserve, dentro il server.py file
    return graph
