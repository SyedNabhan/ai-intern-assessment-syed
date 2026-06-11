# Notes

---


first day , just setup stuff. installed python , git , vscode
learned what RAG is and what LangGraph is , watched some videos
got API keys from Groq , HuggingFace , Langfuse
made a github repo , added .gitignore so i dont accidentally push my keys
read the articles they gave me about chunking and MCP , understood maybe 60% of it

---


started task 1 , had to build a RAG system over FastAPI docs
first thing was writing ingest.py to process all the documents
took me a while to understand what chunking even means practically
ran it on the FastAPI docs folder , processed 153 files
ChromaDB got created with 3869 chunks , felt good seeing that number
pushed to github

---


built the actual retrieval part — combining vector search with BM25
didnt fully understand BM25 at first , had to read about it again
added a reranker on top , results got noticeably better
added query rewriting so if someone asks a vague question it gets rewritten first
tested it manually with a bunch of questions , working well
the system correctly says "i dont know" when the answer isnt in the docs which was important

---


tried to run RAGAS evaluation — crashed immediately
turns out python 3.14 doesnt support RAGAS
installed python 3.11 , switched the interpreter in vscode , RAGAS worked
new problem — groq kept hitting token limits halfway through evaluation so no scores
fixed it by using a smaller model just for evaluation
finally got scores , saved the report

---


built the streamlit UI
added chat history , citation panel , chunk viewer on the side
added a comparison toggle for vector only vs hybrid+reranker
added a logo just to make it look nicer
wrote the 25 question golden set , this took way longer than i expected
task 1 done

---



started task 2 , building a research agent with LangGraph
spent most of the day just understanding how LangGraph works
graphs , nodes , edges , state — took time to click
logged into langfuse , got the keys , set up observability
built the skeleton graph with empty nodes just to see the state flow
traces showing up in langfuse dashboard , that was satisfying

---



wrote all 4 tools — web search , scraper , RAG query , report formatter
filled in the actual node logic
hit a lot of errors today
pydantic doesnt work with langgraph state , switched to TypedDict
the llama model i was using got shut down , switched to a newer one
chromadb collection wasnt found , had to fix the working directory
graph was looping forever , fixed the stop condition
duckduckgo library stopped working , replaced it with ddgs
langfuse api changed , had to use a different method
eventually all 3 test runs worked and got saved

---



built the streamlit UI for task 2
spent most of the day debugging why agent.py wouldnt import in the app
turns out uv run uses a different python than the system python
removed venv , switched to system python , import worked
then chromadb version was wrong , upgraded it
then httpx had a conflict , pinned an older version
then the RAG import path was wrong , fixed that too
wrote 18 guardrail tests
the gibberish detection was catching real gibberish as "too vague" because i had the checks in wrong order , fixed the order
all 18 tests passing
task 2 done

---



reviewed both tasks
made sure everything was clean and committed
re-ran the 3 example runs and saved them
checked langfuse , all traces there

---



started task 3 , building an MCP server
MCP was new to me , read the docs , watched a video
chose github issues manager because i already had the token
wrote github_client.py to talk to the github API
kept getting 404 because the repo was private and the token had no write permission
made repo public , regenerated token with the right permissions
built server.py with 6 tools using FastMCP
write tools need confirm=True so nothing happens by accident
created some test issues in the repo so theres actual data

---



connected the server to Cursor Desktop via the mcp config file
restarted cursor desktop and in agents , i tested and saw that my mcp is conmnected ,
ran the sample prompts , took screenshots
wrote the README with the config block so someone else can set it up
task 3 done

---



task 4 , the final one , had to connect everything together
built orchestrator.py that calls task 1 , task 2 , and task 3 one after another
then groq combines all the results into one answer
built the chat UI with a live panel showing whats happening at each step
cost and latency shows under every answer
added a button to create a github issue from any question
tested with 3 different query types , all working
even random questions like football work , RAG says not found then agent searches the web
wrote the architecture diagram and limitations file
all 4 tasks done and pushed
