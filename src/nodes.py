import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.state import GraphState

# --- LAZY LOADING PATTERN ---
# We define a function to get the data, but we don't RUN it immediately.
def get_retriever():
    PERSIST_DIR = "./chroma_db"
    embedding = OpenAIEmbeddings()
    
    # Check if DB exists
    if os.path.exists(PERSIST_DIR) and os.path.isdir(PERSIST_DIR):
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
        
        # --- SAFETY CHECK: IS IT EMPTY? ---
        # If the DB exists but has 0 items, it's corrupt. Delete and rebuild.
        # (Note: _collection.count() is a ChromaDB specific method)
        if vectorstore._collection.count() == 0:
            print("--- VECTOR STORE EMPTY/CORRUPT. REBUILDING... ---")
            # Fall through to the creation logic below...
        else:
            print("--- LOADING EXISTING VECTOR STORE ---")
            return vectorstore.as_retriever()

    # --- CREATION LOGIC (Runs if missing OR empty) ---
    print("--- BUILDING VECTOR STORE ---")
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=doc_splits, 
        collection_name="rag-chroma", 
        embedding=embedding, 
        persist_directory=PERSIST_DIR
    )
    return vectorstore.as_retriever()


# Initialize objects
# Note: We call get_retriever() here, but because it's now a function,
# Python handles the memory management better for Streamlit.
retriever = get_retriever()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- NODES (Keep these exactly the same) ---

def _doc_meta(d, idx: int) -> dict:
    """Metadata-only representation of a Document for safe tracing/logging."""
    meta = getattr(d, "metadata", None) or {}
    # Common keys from loaders include 'source' (URL/path). Keep it flexible.
    out = {"index": idx}
    if isinstance(meta, dict):
        # Shallow copy of metadata only (no page_content)
        out.update(meta)
    return out

def retrieve(state: GraphState):
    print(f"---RETRIEVE (Attempt: {state.get('retry_count', 0)})---")
    question = state["question"]
    documents = retriever.invoke(question)
    retrieved_docs_meta = [_doc_meta(d, idx) for idx, d in enumerate(documents)]
    return {"documents": documents, "question": question, "retrieved_docs_meta": retrieved_docs_meta}

def generate(state: GraphState):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    used_docs_meta = [_doc_meta(d, idx) for idx, d in enumerate(documents)]
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
    )
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": documents, "question": question})
    return {"generation": generation, "used_docs_meta": used_docs_meta}

def grade_documents(state: GraphState):
    print("---CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)
    
    system = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as 'yes'. 
    Otherwise grade it as 'no'.

    Output MUST be exactly one token: 'yes' or 'no'. No punctuation, no explanation."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ])
    grader_llm = grade_prompt | llm | StrOutputParser()
    
    filtered_docs = []
    relevant_found = False
    doc_grades = []
    
    for idx, d in enumerate(documents):
        score = grader_llm.invoke({"question": question, "document": d.page_content})
        normalized = (score or "").strip().lower()
        # Be tolerant to minor punctuation/formatting despite instructions.
        first_token = (normalized.split()[0] if normalized else "").strip(".,;:!?\"'")
        passed = first_token == "yes"
        doc_grades.append(
            {
                "index": idx,
                "passed": passed,
                "grade": first_token or normalized,
                # metadata-only (no text)
                **_doc_meta(d, idx),
            }
        )
        if passed:
            filtered_docs.append(d)
            relevant_found = True
            
    if relevant_found:
        return {"documents": filtered_docs, "doc_grades": doc_grades}
    else:
        return {"documents": [], "retry_count": retry_count + 1, "doc_grades": doc_grades}

def rewrite_query(state: GraphState):
    print("---REWRITE QUERY---")
    question = state["question"]
    msg = [
        ("system", "You are a search query optimizer. Look at the input and try to reason about the underlying semantic intent / meaning."),
        ("human", f"Here is the initial question: \n\n {question} \n Formulate an improved question.")
    ]
    rewriter = ChatPromptTemplate.from_messages(msg) | llm | StrOutputParser()
    better_question = rewriter.invoke({})
    return {"question": better_question}