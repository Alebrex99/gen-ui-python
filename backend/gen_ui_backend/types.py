from typing import List, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel

#il tipo qui definito coincide con un oggetto
# {"input": List[Union[HumanMessage, AIMessage, SystemMessage]]}
class ChatInputType(BaseModel):
    input: List[Union[HumanMessage, AIMessage, SystemMessage]]
    #definisce direttamente i tipi di INPUT
    # che la nostra CHAIN può accettare
