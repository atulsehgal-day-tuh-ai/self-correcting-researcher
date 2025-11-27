import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.state import GraphState

# --- CONFIGURATION ---
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = OpenAIEmbeddings()

# --- OPTIMIZED LOADING LOGIC ---
if os.path.exists(PERSIST_DIR) and os.path.isdir(PERSIST_DIR):
    print("--- LOADING EXISTING VECTOR STORE (NO API COST) ---")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=EMBEDDING_MODEL
    )
else:
    print("--- CREATING NEW VECTOR STORE (API CALLS) ---")
    # 1. Load Data
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()
    
    # 2. Split Data
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs)
    
    # 3. Create & Save to Disk
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=EMBEDDING_MODEL,
        persist_directory=PERSIST_DIR
    )

retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# --- NODES ---

def retrieve(state: GraphState):
    """Retrieve documents from VectorDB"""
    print(f"---RETRIEVE (Attempt: {state.get('retry_count', 0)})---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    """Generate answer using RAG"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Simple RAG Chain
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
    )
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": documents, "question": question})
    return {"generation": generation}

def grade_documents(state: GraphState):
    """
    Determines if the retrieved documents are relevant to the question.
    If any document is not relevant, we will set a flag to rewrite the query.
    """
    print("---CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)
    
    # LLM Grader
    system = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as 'yes'. 
    Otherwise grade it as 'no'."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ])
    grader_llm = grade_prompt | llm | StrOutputParser()
    
    # Check each doc
    filtered_docs = []
    relevant_found = False
    
    for d in documents:
        score = grader_llm.invoke({"question": question, "document": d.page_content})
        if "yes" in score.lower():
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            relevant_found = True
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            
    if relevant_found:
        return {"documents": filtered_docs}
    else:
        # No relevant docs found? Increment retry count to trigger rewrite
        return {"documents": [], "retry_count": retry_count + 1}

def rewrite_query(state: GraphState):
    """Rewrite the question to produce a better search query"""
    print("---REWRITE QUERY---")
    question = state["question"]
    
    msg = [
        ("system", "You are a search query optimizer. Look at the input and try to reason about the underlying semantic intent / meaning."),
        ("human", f"Here is the initial question: \n\n {question} \n Formulate an improved question.")
    ]
    
    rewriter = ChatPromptTemplate.from_messages(msg) | llm | StrOutputParser()
    better_question = rewriter.invoke({})
    
    print(f"---QUERY REWRITTEN: {better_question}---")
    return {"question": better_question}