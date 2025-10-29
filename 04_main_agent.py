from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, Sequence, Dict, Any
import operator
from crag_module import execute_crag_workflow_simple
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END

# --- Tool Definitions ---
@tool
def crag_retrieval_tool(query: str) -> str:
    """
    Use this tool for questions about CRAG, Self-RAG, retrieval methods, 
    research papers, or the internal knowledge base.
    """
    print(f"\n[CRAG TOOL ACTIVATED]")
    return execute_crag_workflow_simple(query)

@tool
def web_search_tool(query: str) -> str:
    """
    Use this tool for current events, news, real-time data, or general facts.
    """
    print(f"\n[WEB SEARCH TOOL ACTIVATED]")
    tavily_tool = TavilySearchResults(max_results=3)
    results = tavily_tool.invoke({"query": query})
    formatted = []
    for r in results:
        if isinstance(r, dict) and r.get('content'):
            formatted.append(r['content'])
    return "\n\n".join(formatted) if formatted else "No results found."

# Create tool map
tools_map = {
    "crag_retrieval_tool": crag_retrieval_tool,
    "web_search_tool": web_search_tool
}

class AgentState(TypedDict):
    query: str
    answer: str

def should_use_crag(query: str) -> bool:
    """Determine if query needs CRAG tool."""
    keywords = ['crag', 'self-rag', 'retrieval', 'evaluator', 'decompose', 
                'recompose', 'rag', 'research', 'paper', 'knowledge', 
                'corrective', 'document', 'llm']
    query_lower = query.lower()
    return any(kw in query_lower for kw in keywords)

def route_and_execute(state: AgentState) -> Dict[str, Any]:
    """Route query to appropriate tool and execute."""
    query = state["query"]
    
    try:
        if should_use_crag(query):
            print("\n→ Routing to CRAG retrieval tool...")
            answer = tools_map["crag_retrieval_tool"].invoke({"query": query})
        else:
            print("\n→ Routing to web search tool...")
            answer = tools_map["web_search_tool"].invoke({"query": query})
        
        return {"answer": answer}
    
    except Exception as e:
        error_msg = f"Error executing tool: {str(e)}"
        print(f"\n✗ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"answer": error_msg}

# Build simple graph
workflow = StateGraph(AgentState)
workflow.add_node("route_and_execute", route_and_execute)
workflow.set_entry_point("route_and_execute")
workflow.add_edge("route_and_execute", END)
app = workflow.compile()

def run_query(query: str) -> str:
    """Run a single query through the system."""
    print(f"\n{'#'*70}")
    print(f"# Processing: {query}")
    print(f"{'#'*70}")
    
    try:
        result = app.invoke({"query": query, "answer": ""})
        answer = result.get("answer", "No answer generated")
        
        print(f"\n{'='*70}")
        print("ANSWER:")
        print(f"{'='*70}")
        print(f"\n{answer}\n")
        print(f"{'='*70}\n")
        
        return answer
        
    except Exception as e:
        error_msg = f"Error in query execution: {str(e)}"
        print(f"\n✗ {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" HYBRID CRAG MULTI-AGENT SYSTEM")
    print(" Type 'exit' to quit")
    print("="*70)
    print("\nExample queries:")
    print("  1. What is CRAG and how does it work?")
    print("  2. Explain the decompose-then-recompose algorithm")
    print("  3. What is the capital of France?")
    print()
    
    while True:
        query = input("Query: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!\n")
            break
        
        if not query:
            continue
        
        run_query(query)

