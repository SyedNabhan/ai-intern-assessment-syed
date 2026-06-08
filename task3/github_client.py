import httpx
import os
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.github.com"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

def list_issues(repo: str, state: str = "open", limit: int = 10) -> list[dict]:
    r = httpx.get(f"{BASE}/repos/{repo}/issues", 
                  headers=HEADERS, 
                  params={"state": state, "per_page": limit})
    r.raise_for_status()
    return [{"number": i["number"], "title": i["title"], 
             "state": i["state"], "labels": [l["name"] for l in i["labels"]],
             "url": i["html_url"]} for i in r.json()]

def get_issue(repo: str, number: int) -> dict:
    r = httpx.get(f"{BASE}/repos/{repo}/issues/{number}", headers=HEADERS)
    r.raise_for_status()
    i = r.json()
    return {"number": i["number"], "title": i["title"],
            "body": i["body"], "state": i["state"],
            "labels": [l["name"] for l in i["labels"]],
            "comments": i["comments"], "url": i["html_url"]}

def search_issues(repo: str, query: str) -> list[dict]:
    r = httpx.get(f"{BASE}/repos/{repo}/issues",
                  headers=HEADERS,
                  params={"state": "all", "per_page": 30})
    r.raise_for_status()
    query_lower = query.lower()
    results = [
        {"number": i["number"], "title": i["title"],
         "state": i["state"], "url": i["html_url"]}
        for i in r.json()
        if query_lower in i["title"].lower() 
        or query_lower in (i.get("body") or "").lower()
    ]
    return results[:10]

def create_issue(repo, title, body, labels=None):
    labels = labels or []
    r = httpx.post(f"{BASE}/repos/{repo}/issues",
                   headers=HEADERS,
                   json={"title": title, "body": body, "labels": labels})
    r.raise_for_status()
    i = r.json()
    return {"number": i["number"], "title": i["title"], "url": i["html_url"]}

def add_comment(repo: str, number: int, body: str) -> dict:
    r = httpx.post(f"{BASE}/repos/{repo}/issues/{number}/comments",
                   headers=HEADERS, json={"body": body})
    r.raise_for_status()
    return {"id": r.json()["id"], "url": r.json()["html_url"]}

def apply_labels(repo: str, number: int, labels: list[str]) -> dict:
    r = httpx.put(f"{BASE}/repos/{repo}/issues/{number}/labels",
                  headers=HEADERS, json={"labels": labels})
    r.raise_for_status()
    return {"labels": [l["name"] for l in r.json()]}