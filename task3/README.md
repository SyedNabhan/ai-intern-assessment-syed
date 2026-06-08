\# Task 3 — GitHub Issues MCP Server



A \*\*Model Context Protocol (MCP) server\*\* that gives AI clients (Cursor, MCP Inspector, Claude Desktop) the ability to manage GitHub Issues as native tools. The server runs locally and communicates over \*\*stdio transport\*\* — the AI client launches it as a child process and calls its tools directly.



\---



\## What It Does



The server exposes 6 tools over MCP:



| Tool | Type | Description |

|---|---|---|

| `list\_github\_issues` | Read | List open/closed issues with state and labels |

| `get\_github\_issue` | Read | Fetch full details of a single issue by number |

| `search\_github\_issues` | Read | Search issues by keyword across title and body |

| `create\_github\_issue` | \*\*Write\*\* | Create a new issue (requires `confirm=True`) |

| `add\_github\_comment` | \*\*Write\*\* | Add a comment to an issue (requires `confirm=True`) |

| `apply\_github\_labels` | \*\*Write\*\* | Apply labels to an issue (requires `confirm=True`) |



Write tools are gated behind a `confirm=True` parameter to prevent accidental mutations when the user only asked a read question.



\---



\## Project Structure



```

task3/

├── .env                    ← GITHUB\_TOKEN + GITHUB\_REPO

├── github\_client.py        ← httpx wrapper for GitHub REST API

├── server.py               ← FastMCP server with 6 tools

└── create\_test\_issues.py   ← script to seed test issues

```



\---



\## Setup



\### 1. Clone and navigate



```bash

cd task3

```



\### 2. Install dependencies



```bash

pip install mcp httpx python-dotenv

```



\### 3. Create `.env`



```env

GITHUB\_TOKEN=github\_pat\_your\_token\_here

GITHUB\_REPO=your-username/your-repo

```



\### 4. Get a GitHub Fine-Grained Token



1\. Go to \*\*github.com → Settings → Developer settings → Personal access tokens → Fine-grained tokens\*\*

2\. Click \*\*Generate new token\*\*

3\. Set \*\*Repository access\*\* → Only select repositories → choose your repo

4\. Under \*\*Repository permissions\*\*, set \*\*Issues → Read and write\*\*

5\. Generate and copy the token into `.env`



\### 5. Run the server



```bash

python server.py

```



The server will hang silently — that's correct. It's listening on stdio for an MCP client to connect.



\---



\## Connecting Clients



\### MCP Inspector (for testing)



```bash

npx @modelcontextprotocol/inspector \\

&#x20; C:\\Users\\<you>\\AppData\\Local\\Programs\\Python\\Python311\\python.exe \\

&#x20; path\\to\\task3\\server.py

```



In the Inspector UI:

1\. Add environment variables: `GITHUB\_TOKEN` and `GITHUB\_REPO`

2\. Click \*\*Connect\*\*

3\. Go to the \*\*Tools\*\* tab to call each tool



\### Cursor



Create `\~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project-level):



```json

{

&#x20; "mcpServers": {

&#x20;   "github-issues": {

&#x20;     "command": "C:\\\\Users\\\\<you>\\\\AppData\\\\Local\\\\Programs\\\\Python\\\\Python311\\\\python.exe",

&#x20;     "args": \["D:\\\\path\\\\to\\\\task3\\\\server.py"],

&#x20;     "env": {

&#x20;       "GITHUB\_TOKEN": "your\_token\_here",

&#x20;       "GITHUB\_REPO": "your-username/your-repo"

&#x20;     }

&#x20;   }

&#x20; }

}

```



Open Cursor → Settings → MCP → verify `github-issues` shows a green dot. Then open Agent chat (Ctrl+L) and type natural language prompts.



\## Sample Prompts



Try these in Cursor Agent or any connected MCP client:



```

1\. List all open issues in my GitHub repository

2\. Search for issues mentioning "RAG" or "retrieval"

3\. Get the full details of issue #11

4\. Create a new issue titled "Task 3 MCP server complete" — go ahead and confirm

5\. Add a comment to issue #11 saying "Verified on Day 11" and confirm it

6\. Apply the label "bug" to issue #8 and confirm

```



\---



\## The `confirm=True` Safety Pattern



Write tools require an explicit `confirm=True` parameter. Without it:



```json

{"error": "confirm=True required to create an issue. Re-call with confirm=True to proceed."}

```



This prevents the AI from accidentally creating issues, posting comments, or modifying labels when the user only asked a read question. The model must receive explicit user intent before any write operation executes.



\---



\## Tested Clients



| Client | Status |

|---|---|

| MCP Inspector v0.22.0 | ✅ All 6 tools verified |

| Cursor (Composer 2.5) | ✅ All 6 tools verified |

| Claude Desktop | ✅ Server connects (Pro plan required for tool UI) |



\---



\## Architecture



```

MCP Client (Cursor / Inspector)

&#x20;       ↓  stdio (stdin/stdout)

&#x20;   server.py  (FastMCP)

&#x20;       ├── list\_github\_issues()

&#x20;       ├── get\_github\_issue()

&#x20;       ├── search\_github\_issues()

&#x20;       ├── create\_github\_issue()     ← confirm=True required

&#x20;       ├── add\_github\_comment()      ← confirm=True required

&#x20;       └── apply\_github\_labels()     ← confirm=True required

&#x20;             ↓

&#x20;       github\_client.py  (httpx)

&#x20;             ↓

&#x20;       api.github.com

```

