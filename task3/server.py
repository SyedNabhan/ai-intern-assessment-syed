import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from github_client import (
    list_issues, get_issue, search_issues,
    create_issue, add_comment, apply_labels
)
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
REPO = os.getenv("GITHUB_REPO")

mcp = FastMCP("GitHub Issues Manager")

@mcp.tool()
def list_github_issues(state: str = "open", limit: int = 10) -> list[dict]:
    """
    Lists issues from the assessment GitHub repository.
    Use this to get an overview of open or closed issues.

    Input:
      state (str): "open", "closed", or "all". Default: "open"
      limit (int): max issues to return, 1–30. Default: 10

    Output:
      list of {number, title, state, labels, url}

    Example call: list_github_issues(state="open", limit=5)
    """
    if state not in ("open", "closed", "all"):
        return [{"error": "state must be 'open', 'closed', or 'all'"}]
    if not 1 <= limit <= 30:
        return [{"error": "limit must be between 1 and 30"}]
    return list_issues(REPO, state, limit)


@mcp.tool()
def get_github_issue(number: int) -> dict:
    """
    Fetches full details of a single GitHub issue by number.

    Input:
      number (int, required): the issue number, e.g. 42

    Output:
      {number, title, body, state, labels, comments, url}

    Example call: get_github_issue(number=1)
    """
    if number < 1:
        return {"error": "Issue number must be a positive integer"}
    try:
        return get_issue(REPO, number)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def search_github_issues(query: str) -> list[dict]:
    """
    Searches issues in the repository by keyword.

    Input:
      query (str, required): search terms, e.g. "bug login page"

    Output:
      list of {number, title, state, url} — up to 10 results

    Example call: search_github_issues(query="RAG pipeline error")
    """
    if not query.strip():
        return [{"error": "query cannot be empty"}]
    return search_issues(REPO, query)


@mcp.tool()
def create_github_issue(title: str, body: str, 
                        labels: list[str] = [], 
                        confirm: bool = False) -> dict:
    """
    Creates a new issue in the repository. WRITE OPERATION — requires confirm=True.

    Input:
      title (str, required): issue title
      body (str, required): issue description in markdown
      labels (list[str]): optional label names to apply
      confirm (bool, required): must be True to proceed. Prevents accidental writes.

    Output:
      {number, title, url} of the created issue, or error if confirm=False

    Example call: create_github_issue(title="Bug: login fails", body="Steps to reproduce...", confirm=True)

    Side effect: creates a real GitHub issue. Cannot be undone via this tool.
    """
    if not confirm:
        return {"error": "confirm=True required to create an issue. Re-call with confirm=True to proceed."}
    if not title.strip():
        return {"error": "title cannot be empty"}
    if not body.strip():
        return {"error": "body cannot be empty"}
    return create_issue(REPO, title, body, labels)


@mcp.tool()
def add_github_comment(number: int, body: str, confirm: bool = False) -> dict:
    """
    Adds a comment to an existing GitHub issue. WRITE OPERATION — requires confirm=True.

    Input:
      number (int, required): issue number to comment on
      body (str, required): comment text in markdown
      confirm (bool, required): must be True to proceed

    Output:
      {id, url} of the created comment

    Example call: add_github_comment(number=5, body="Fixed in PR #12", confirm=True)

    Side effect: posts a public comment. Cannot be deleted via this tool.
    """
    if not confirm:
        return {"error": "confirm=True required to add a comment."}
    if not body.strip():
        return {"error": "body cannot be empty"}
    return add_comment(REPO, number, body)


@mcp.tool()
def apply_github_labels(number: int, labels: list[str], confirm: bool = False) -> dict:
    """
    Applies labels to an existing GitHub issue. WRITE OPERATION — requires confirm=True.
    Replaces all existing labels with the provided list.

    Input:
      number (int, required): issue number
      labels (list[str], required): label names to apply, e.g. ["bug", "priority-high"]
      confirm (bool, required): must be True to proceed

    Output:
      {labels} list of applied label names

    Example call: apply_github_labels(number=3, labels=["bug", "task-2"], confirm=True)

    Side effect: overwrites existing labels on the issue.
    """
    if not confirm:
        return {"error": "confirm=True required to apply labels."}
    if not labels:
        return {"error": "labels list cannot be empty"}
    return apply_labels(REPO, number, labels)


if __name__ == "__main__":
    mcp.run(transport="stdio")