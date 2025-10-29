from dotenv import load_dotenv
load_dotenv()

import torch
import re
from typing import List, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

# --- Load all necessary components once ---
print("Loading CRAG module components...")
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
evaluator_tokenizer = AutoTokenizer.from_pretrained("./models/retrieval_evaluator")
evaluator_model = AutoModelForSequenceClassification.from_pretrained("./models/retrieval_evaluator")
generator_llm = ChatOllama(model="llama3", temperature=0)
web_search_tool = TavilySearchResults(k=3)
print("✓ CRAG module components loaded.\n")


def split_into_knowledge_strips(text: str, min_length: int = 50) -> List[str]:
    """Split document into knowledge strips (sentences or small semantic units)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    strips = []
    current_strip = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_strip) < min_length:
            current_strip += " " + sentence if current_strip else sentence
        else:
            strips.append(current_strip)
            current_strip = sentence
    
    if current_strip:
        strips.append(current_strip)
    
    return strips


def knowledge_refinement(
    query: str, 
    documents: List, 
    evaluator_model, 
    evaluator_tokenizer, 
    threshold: float = 0.15  # LOWERED from 0.3
) -> Tuple[str, float]:
    """
    Core CRAG algorithm: Decompose-then-Recompose.
    Returns: (refined_context, avg_confidence_score)
    """
    knowledge_strips = []
    
    for doc in documents:
        doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        strips = split_into_knowledge_strips(doc_text)
        
        for strip in strips:
            if not strip.strip() or len(strip.split()) < 5:
                continue
                
            inputs = evaluator_tokenizer(
                query, 
                strip, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                logits = evaluator_model(**inputs).logits
                scores = torch.softmax(logits, dim=1).squeeze()
                
            relevance_score = scores[0].item()
            
            if relevance_score > threshold:
                knowledge_strips.append((strip, relevance_score))
    
    knowledge_strips.sort(key=lambda x: x[1], reverse=True)
    top_strips = knowledge_strips[:10]  # INCREASED from 7
    
    avg_confidence = sum(score for _, score in top_strips) / len(top_strips) if top_strips else 0.0
    refined_context = "\n\n".join([strip for strip, _ in top_strips])
    
    return refined_context, avg_confidence


def rewrite_query_for_search(query: str, llm) -> str:
    """Rewrite user query into web search keywords."""
    prompt = f"""Extract 2-4 keywords from the question for web search. Output ONLY the keywords separated by spaces.

Examples:
Question: What is Henry Feilden's occupation?
Keywords: Henry Feilden occupation

Question: How does photosynthesis work in plants?
Keywords: photosynthesis plants process

Question: {query}
Keywords:"""
    
    try:
        response = llm.invoke(prompt)
        keywords = response.content.strip()
        return keywords if keywords and len(keywords) > 0 else query
    except:
        return query


def verify_answer_attribution(answer: str, context: str, llm) -> dict:
    """Post-generation verification to detect hallucinations."""
    prompt = f"""You are a fact-checker. Verify if the answer is fully supported by the context.

Context:
{context}

Answer:
{answer}

Are there any claims in the answer NOT supported by the context? Respond with:
- "VERIFIED" if all claims are supported
- List specific unsupported claims if any exist

Response:"""
    
    try:
        verification = llm.invoke(prompt)
        result = verification.content.strip()
        is_verified = "VERIFIED" in result.upper()
        
        return {
            "is_verified": is_verified,
            "details": result,
            "has_hallucination": not is_verified
        }
    except Exception as e:
        return {
            "is_verified": False,
            "details": f"Verification failed: {str(e)}",
            "has_hallucination": True
        }


def execute_crag_workflow(query: str, verbose: bool = True) -> dict:
    """
    Enhanced CRAG workflow with knowledge refinement and verification.
    Returns: dict with answer, action, confidence, verification
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[CRAG MODULE ACTIVATED]")
        print(f"Query: '{query}'")
        print(f"{'='*60}\n")
    
    if verbose:
        print("→ Step 1: Retrieving documents from knowledge base...")
    retrieved_docs = retriever.invoke(query)
    
    if verbose:
        print("→ Step 2: Evaluating retrieval quality...")
    
    docs_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
    inputs = evaluator_tokenizer(
        query, 
        docs_content, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    )
    
    with torch.no_grad():
        logits = evaluator_model(**inputs).logits
        scores = torch.softmax(logits, dim=1).squeeze()
    
    correct_score = scores[0].item()
    incorrect_score = scores[2].item()
    
    CORRECT_THRESHOLD = 0.40  # LOWERED from 0.45
    INCORRECT_THRESHOLD = 0.50
    
    final_context = ""
    action = ""
    confidence = 0.0
    
    if correct_score > CORRECT_THRESHOLD:
        action = "CORRECT"
        if verbose:
            print(f"→ Decision: {action} (confidence: {correct_score:.3f})")
            print("→ Step 3: Applying knowledge refinement...")
        
        final_context, confidence = knowledge_refinement(
            query, retrieved_docs, evaluator_model, evaluator_tokenizer, threshold=0.15  # LOWERED
        )
        
    elif incorrect_score > INCORRECT_THRESHOLD:
        action = "INCORRECT"
        if verbose:
            print(f"→ Decision: {action} (confidence: {incorrect_score:.3f})")
            print("→ Step 3: Using web search...")
        
        search_query = rewrite_query_for_search(query, generator_llm)
        if verbose:
            print(f"   Rewritten query: '{search_query}'")
        
        web_results = web_search_tool.invoke({"query": search_query})
        web_docs = [
            type('obj', (object,), {'page_content': r.get("content", "")})() 
            for r in web_results if r.get("content")
        ]
        
        final_context, confidence = knowledge_refinement(
            query, web_docs, evaluator_model, evaluator_tokenizer, threshold=0.12  # LOWERED
        )
        
    else:
        action = "AMBIGUOUS"
        if verbose:
            print(f"→ Decision: {action}")
            print("→ Step 3: Combining internal + web search...")
        
        internal_context, internal_conf = knowledge_refinement(
            query, retrieved_docs, evaluator_model, evaluator_tokenizer, threshold=0.15  # LOWERED
        )
        
        search_query = rewrite_query_for_search(query, generator_llm)
        web_results = web_search_tool.invoke({"query": search_query})
        web_docs = [
            type('obj', (object,), {'page_content': r.get("content", "")})() 
            for r in web_results if r.get("content")
        ]
        
        external_context, external_conf = knowledge_refinement(
            query, web_docs, evaluator_model, evaluator_tokenizer, threshold=0.12  # LOWERED
        )
        
        final_context = f"{internal_context}\n\n--- Additional Web Information ---\n\n{external_context}"
        confidence = (internal_conf + external_conf) / 2
    
    if verbose:
        print("→ Step 4: Generating answer...")
    
    # IMPROVED PROMPT
    prompt = f"""You are an expert on Corrective Retrieval Augmented Generation (CRAG), Self-RAG, and related AI research topics. Answer the question using the provided context.

Instructions:
- Provide a detailed, comprehensive answer using information from the context
- Include key concepts, definitions, algorithms, and technical details
- Explain how things work step-by-step when relevant
- Be specific and cite information from the context
- If the context fully answers the question, give a thorough explanation
- Only state lack of information if the context truly doesn't address the question at all

Context:
{final_context}

Question: {query}

Detailed Answer:"""
    
    response = generator_llm.invoke(prompt)
    answer = response.content
    
    if verbose:
        print("→ Step 5: Verifying answer...")
    
    verification = verify_answer_attribution(answer, final_context, generator_llm)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[CRAG MODULE COMPLETE]")
        print(f"Action: {action} | Confidence: {confidence:.3f}")
        print(f"Verification: {'✓ PASSED' if verification['is_verified'] else '✗ FAILED'}")
        print(f"{'='*60}\n")
    
    return {
        "answer": answer,
        "action": action,
        "confidence": confidence,
        "verification": verification,
        "context_used": final_context
    }


def execute_crag_workflow_simple(query: str) -> str:
    """Simple interface that returns just the answer string."""
    result = execute_crag_workflow(query, verbose=True)
    return result["answer"]
