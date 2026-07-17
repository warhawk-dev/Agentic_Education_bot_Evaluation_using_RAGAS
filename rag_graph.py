import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import logging
from typing import Any
from langchain_core.messages import ToolMessage, AIMessage
from langchain.agents.middleware import before_agent, wrap_tool_call, AgentState
from langgraph.runtime import Runtime

logger = logging.getLogger("agentic_rag")

# ── Models ────────────────────────────────────────────────────────────────────

llm        = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

physics_db = Chroma(
    persist_directory="chroma_db/physics",
    embedding_function=embeddings,
)

chemistry_db = Chroma(
    persist_directory="chroma_db/chemistry",
    embedding_function=embeddings,
)

biology_db = Chroma(
    persist_directory="chroma_db/biology",
    embedding_function=embeddings,
)

SUBJECT_STORES = {
    "physics": physics_db,
    "chemistry": chemistry_db,
    "biology": biology_db,
}

# ── PDF indexing helper ───────────────────────────────────────────────────────

def index_pdf(pdf_path: str, subject: str) -> int:
    """
    Reads a PDF and indexes it into the matching ChromaDB store.
    Returns the number of chunks created.
    """
    store = SUBJECT_STORES[subject]

    # Step A: Load all pages from the PDF
    documents = PyPDFLoader(pdf_path).load()

    # Step B: Split pages into smaller overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    # Step C: Add chunks to ChromaDB
    store.add_documents(chunks)
    return len(chunks)

# ── Retriever tools ───────────────────────────────────────────────────────────

@tool
def physics(query: str) -> str:
    """
    Search the Physics textbook for relevant content.
    Use this tool for questions about:
    - Mechanics: motion, velocity, acceleration, force, friction, Newton's laws
    - Energy: kinetic energy, potential energy, work, power, conservation of energy
    - Waves: sound waves, light waves, frequency, amplitude, wavelength
    - Optics: reflection, refraction, lenses, mirrors
    - Electricity & Magnetism: current, voltage, resistance, circuits, magnetic fields
    - Modern Physics: atomic structure, radioactivity, nuclear reactions
    Do NOT use this for chemistry or biology questions.
    """
    docs = physics_db.similarity_search(query, k=3)
    formatted_chunks = []
    for d in docs:
        source = d.metadata.get("source", "Physics.pdf")
        page = d.metadata.get("page", "?")
        text = d.page_content
        formatted_chunks.append(f"[Source: {source} | Page: {page}]\n{text}")
    return "\n\n".join(formatted_chunks)


@tool
def chemistry(query: str) -> str:
    """
    Search the Chemistry textbook for relevant content.
    Use this tool for questions about:
    - Atomic Structure: atoms, electrons, protons, neutrons, orbitals, electron configuration
    - Periodic Table: elements, groups, periods, atomic number, atomic mass
    - Chemical Bonding: ionic bonds, covalent bonds, hydrogen bonds, valence electrons
    - Reactions: chemical equations, balancing, oxidation, reduction, acids, bases
    - Organic Chemistry: carbon compounds, benzene, hydrocarbons, functional groups
    - States of Matter: solids, liquids, gases, phase changes, intermolecular forces
    Do NOT use this for physics or biology questions.
    """
    docs = chemistry_db.similarity_search(query, k=3)
    formatted_chunks = []
    for d in docs:
        source = d.metadata.get("source", "Chemistry.pdf")
        page = d.metadata.get("page", "?")
        text = d.page_content
        formatted_chunks.append(f"[Source: {source} | Page: {page}]\n{text}")
    return "\n\n".join(formatted_chunks)


@tool
def biology(query: str) -> str:
    """
    Search the Biology textbook for relevant content.
    Use this tool for questions about:
    - Cell Biology: cell structure, cell membrane, mitochondria, nucleus, organelles
    - Genetics: DNA, RNA, genes, chromosomes, heredity, mutations, genetic disorders
    - Evolution: natural selection, adaptation, Charles Darwin, species, fossils
    - Human Body: organ systems, digestion, respiration, circulation, nervous system
    - Plants: photosynthesis, chlorophyll, plant cells, transpiration, reproduction
    - Ecology: food chains, ecosystems, biodiversity, population, environment
    Do NOT use this for physics or chemistry questions.
    """
    docs = biology_db.similarity_search(query, k=3)
    formatted_chunks = []
    for d in docs:
        source = d.metadata.get("source", "Biology.pdf")
        page = d.metadata.get("page", "?")
        text = d.page_content
        formatted_chunks.append(f"[Source: {source} | Page: {page}]\n{text}")
    return "\n\n".join(formatted_chunks)


all_tools = [physics, chemistry, biology]

# ── Structured output schemas ─────────────────────────────────────────────────

class GradeResult(BaseModel):
    """Relevance grade for a set of retrieved passages against a question."""
    is_relevant: bool = Field(description="Whether the passages are relevant enough to answer the question.")

class FinalAnswer(BaseModel):
    """Final structured answer generated from retrieved passages."""
    answer: str = Field(description="The answer to the user's question, based only on the passages provided.")
    summary: str = Field(description="A one-sentence summary of the answer.")
    source: str = Field(description="The PDF filename the answer was drawn from, e.g. Physics.pdf.")
    page: str = Field(description="The page number(s) the answer was drawn from.")

#  GUARDRAILS  —  a simple prompt-injection filter
#  1) DIRECT injection  -> the user types the attack straight into the chat box. We catch this by checking the QUESTION before it ever reaches the model.
#  2) INDIRECT injection -> someone hides the attack inside an uploaded PDF. trustworthy textbook content. We catch this by checking the TOOL RESULT
MESSAGE_FOR_BLOCKED_QUESTION = "This request couldn't be processed. Please rephrase your question."
MESSAGE_FOR_BLOCKED_PDF_CONTENT = "[Retrieved content withheld: flagged as a potential prompt injection attempt.]"
INJECTION_PHRASES = [
    # Trying to override or erase earlier instructions
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore the above instructions",
    "disregard previous instructions",
    "forget everything above",
    "override your instructions",
    "new instructions:",
    "system prompt",
    "reveal your prompt",
    "reveal your system prompt",
    "you are now a",
    "you are now an",
    "act as a",
    "act as an",
    "pretend you are",
    "pretend to be",
    "jailbreak",
    "do anything now",
    "without any restrictions",
]

def looks_like_prompt_injection(text: str)->bool:
    """The main guardrail check: True if 'text' contains any of the known injection phrases"""
    if not text or not text.strip():
        return False
    lowercase_text = text.lower()
    is_suspicious = any(phrase in lowercase_text for phrase in INJECTION_PHRASES) 

    if is_suspicious:
        logger.warning(f"Blocked possible prompt injection: {text[:100]!r}")

    return is_suspicious

@before_agent(can_jump_to=["end"])
def block_direct_injection(state: AgentState, runtime: Runtime)-> dict[str, Any] | None:
    """Runs once, right when the agent starts - before the question reaches the model at all"""
    if not state["messages"]:
        return None
    first_message = state["messges"][0]
    if first_message.type != "human":
        return None
    
    question_text = first_message.content
    if not isinstance(question_text, str):
        question_text = str(question_text)

    if looks_like_prompt_injection(question_text):
        return{
            "messages": [{"role": "assistant", "content": MESSAGE_FOR_BLOCKED_QUESTION}],
            "jump_to": "end",
        }
    return None

@wrap_tool_call
def block_indirect_injection(request, handler):
    "checks the result of the tool for INDIRECT injection before that text is allowed back into the agent's conversation"
    tool_result = handler(request) #gets the normal result from the tool
    result_text = tool_result.content if isinstance(tool_result, ToolMessage) else None
    if isinstance(result_text, str) and looks_like_prompt_injection(result_text):
        return ToolMessage(content=MESSAGE_FOR_BLOCKED_PDF_CONTENT, tool_call_id= request.tool_call["id"],)
    return tool_result


# ── Agent ─────────────────────────────────────────────────────────────────────

agent = create_agent(
    model=llm,
    tools=all_tools,
    system_prompt="You are a retrieval assistant. You MUST use the provided tools to search for information. NEVER generate answers from your own knowledge.",
    middleware=[block_direct_injection, block_indirect_injection],
)

# ── RAG State ─────────────────────────────────────────────────────────────────

class RAGState(BaseModel):
    question:       str   = Field(default="")
    retrieved_text: str   = Field(default="")
    is_relevant:    bool  = Field(default=False)
    answer:         str   = Field(default="")
    summary:        str   = Field(default="")
    source:         str   = Field(default="")
    page:           str   = Field(default="")
    retry_count:    int   = Field(default=0)

# ── Graph nodes ───────────────────────────────────────────────────────────────

def retrieve_node(state: RAGState) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": state.question}]})
    
    # Find the ToolMessage explicitly, rather than assuming its position
    from langchain_core.messages import ToolMessage
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    retrieved_text = tool_messages[-1].content if tool_messages else ""
    
    return {"retrieved_text": retrieved_text}

def grade_node(state: RAGState) -> dict:
    grade_chain = (
        PromptTemplate.from_template(
            "Question: {question}\n\n"
            "Passages:\n{passages}\n\n"
            "Do these passages contain specific information that directly answers the question? "
            "Answer true only if the passages explicitly contain the answer. "
            "If the passages are about a completely different topic, answer false."
        )
        | llm.with_structured_output(GradeResult)
    )
    try:
        result = grade_chain.invoke({
            "question": state.question,
            "passages": state.retrieved_text[:1500],
        })
        return {"is_relevant": result.is_relevant}
    except Exception:
        # If the model fails to produce valid structured output, default to "not relevant"
        # so the graph safely falls through to rephrase/retry rather than crashing.
        return {"is_relevant": False}


def rephrase_node(state: RAGState) -> dict:
    rephrase_chain = (
        PromptTemplate.from_template(
            "Rewrite this question to be more specific for a textbook search.\n"
            "Return ONLY the rewritten question.\n\n"
            "Original: {question}"
        )
        | llm
        | StrOutputParser()
    )
    new_question = rephrase_chain.invoke({"question": state.question})
    return {"question": new_question, "retry_count": state.retry_count + 1}


def generate_node(state: RAGState) -> dict:
    # If not relevant after retrying, return not found
    if not state.is_relevant:
        return {
            "answer": "No relevant content found in the uploaded PDFs for this question.",
            "summary": "",
            "source": "",
            "page": "",
        }

    generate_chain = (
        PromptTemplate.from_template(
            "Answer the question using only the passages below. If the passages include a "
            "[Source: ... | Page: ...] tag, use it to fill in the source and page fields.\n\n"
            "Question: {question}\n\n"
            "Passages:\n{passages}"
        )
        | llm.with_structured_output(FinalAnswer)
    )
    try:
        result = generate_chain.invoke({
            "question": state.question,
            "passages": state.retrieved_text,
        })
        return {
            "answer": result.answer,
            "summary": result.summary,
            "source": result.source,
            "page": result.page,
        }
    except Exception:
        # Model failed to produce valid structured output (e.g. returned plain text
        # instead of a proper tool call). Fail safely instead of crashing the request.
        return {
            "answer": "Sorry, something went wrong while generating an answer. Please try rephrasing your question.",
            "summary": "",
            "source": "",
            "page": "",
        }

# ── Routing ───────────────────────────────────────────────────────────────────

def after_grade(state: RAGState) -> str:
    if state.is_relevant or state.retry_count >= 1:
        return "generate"
    else:
        return "rephrase"

graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("grade",    grade_node)
graph.add_node("rephrase", rephrase_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", after_grade, {"generate": "generate", "rephrase": "rephrase"})
graph.add_edge("rephrase", "retrieve")
graph.add_edge("generate", END)

rag_app = graph.compile()