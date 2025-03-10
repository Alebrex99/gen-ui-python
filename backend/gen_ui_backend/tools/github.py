import os
from typing import Dict, Union

import requests
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

#https://python.langchain.com/docs/how_to/tool_calling/
#class usabile per definire i campi richiesti. infatti questi sono richiesti da GITHUB API per fetch dati dalla REPO
class GithubRepoInput(BaseModel):
    owner: str = Field(..., description="The name of the repository owner.") #proprietario repo; ... -> ellipsys indica campo required, perchè indica tutto quello che c'è in classe
    repo: str = Field(..., description="The name of the repository.") #nome del repository


@tool("github-repo", args_schema=GithubRepoInput, return_direct=True) #args_schema: schema per i parametri di input; return_redirect: se Treu, restituzione result tool all'utente direttamente
def github_repo(owner: str, repo: str) -> Union[Dict, str]:
    #interrogando il github API, ritorniamo un dizionario (dati API ci fonrisce), se non ci sono errori, altrimenti una stringa
    """Get information about a GitHub repository."""
    if not os.environ.get("GITHUB_TOKEN"):
        raise ValueError("Missing GITHUB_TOKEN secret.")

    #PER CAPIRE QUESTA PARTE: GITHUB REPOSITORY API: https://docs.github.com/en/rest/repos/repos?apiVersion=2022-11-28#list-repository-tags
    #impostazione HTTP header + payload
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    url = f"https://api.github.com/repos/{owner}/{repo}"

    #REQUEST WRAPPING: https://requests.readthedocs.io/en/latest/user/quickstart/#make-a-request
    #si tenta di fare una GET request
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() #lancia un 404 error HTTP se la richiesta non è andata a buon fine
        repo_data = response.json() #prendi i datiin json
        return {
            "owner": owner,
            "repo": repo,
            "description": repo_data.get("description", ""),
            "stars": repo_data.get("stargazers_count", 0),
            "language": repo_data.get("language", ""),
        }
    except requests.exceptions.RequestException as err:
        print(err)
        return "There was an error fetching the repository. Please check the owner and repo names."
