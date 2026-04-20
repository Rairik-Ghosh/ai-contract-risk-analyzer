
"""
capstone_streamlit.py — AI Contract Risk Analyzer
Run: streamlit run capstone_streamlit.py
"""
from datetime import datetime

import streamlit as st
import uuid
import os
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, state
from langgraph.checkpoint.memory import MemorySaver
DOMAIN_NAME = "AI Contract Risk Analyzer"
DOMAIN_DESCRIPTION = "Analyzes contracts, detects risks, and explains issues using AI"
load_dotenv()

st.set_page_config(page_title="AI Contract Risk Analyzer", page_icon="🤖", layout="centered")
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.stChatInput {
    position: fixed;
    bottom: 20px;
    width: 70%;
}
</style>
""", unsafe_allow_html=True)
st.title("🤖 AI Contract Risk Analyzer")
st.caption("Analyzes contracts, detects risks, and explains issues using AI")

# ── Load models and KB (cached) ───────────────────────────
DOCUMENTS = [
    {"id": "doc_001", "topic": "Confidentiality Clause", "text": "A confidentiality clause ensures that sensitive information shared between parties is not disclosed."},
    {"id": "doc_002", "topic": "Termination Clause", "text": "Defines how a contract can be ended and under what conditions."},
    {"id": "doc_003", "topic": "Liability Clause", "text": "Defines responsibility for damages. Unlimited liability is high risk."},
    {"id": "doc_004", "topic": "Payment Terms", "text": "Defines payment timelines, penalties, and methods."},
    {"id": "doc_005", "topic": "Non-Compete Clause", "text": "Restricts competition within time and region."},
    {"id": "doc_006", "topic": "Dispute Resolution", "text": "Defines how disputes are handled."},
    {"id": "doc_007", "topic": "Intellectual Property", "text": "Defines ownership of created work."},
    {"id": "doc_008", "topic": "Force Majeure", "text": "Covers unforeseen events like disasters."},
    {"id": "doc_009", "topic": "Breach and Penalties", "text": "Defines penalties for contract violation."},
    {"id": "doc_010", "topic": "Governing Law", "text": "Defines which laws apply to the contract."}
]
def load_agent():
    os.environ["GROQ_API_KEY"] = "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"
    llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    try: client.delete_collection("capstone_kb")
    except: pass
    collection = client.create_collection("capstone_kb")

    # TODO: Copy your DOCUMENTS list here
   
    texts = [d["text"] for d in DOCUMENTS]
    collection.add(documents=texts, embeddings=embedder.encode(texts).tolist(),
                   ids=[d["id"] for d in DOCUMENTS],
                   metadatas=[{"topic": d["topic"]} for d in DOCUMENTS])

    # TODO: Copy your CapstoneState, node functions, and graph assembly here
    # ===== STATE =====
    class CapstoneState(TypedDict):
        question: str
        document_text: str
        extracted_clauses: dict
        messages: List[dict]
        route: str
        retrieved: str
        sources: List[str]
        tool_result: str
        answer: str
        risk_score: float
        faithfulness: float
        eval_retries: int


    # ===== NODES =====

    def memory_node(state: CapstoneState):
        messages = state.get("messages", [])
        messages.append({"role": "user", "content": state["question"]})
        messages = messages[-6:]
        return {"messages": messages}


    def router_node(state: CapstoneState):
        q = state["question"].lower()
        if "risk" in q or "analyze" in q or "safe" in q:
            return {"route": "tool"}
        elif "what" in q or "explain" in q:
            return {"route": "retrieve"}
        else:
            return {"route": "retrieve"}


    def retrieval_node(state: CapstoneState):
        q_emb = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)

        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]

        context = "---".join(
            f"[{topics[i]}]{chunks[i]}" for i in range(len(chunks))
        )

        return {"retrieved": context, "sources": topics}


    def skip_node(state: CapstoneState):
        return {"retrieved": "", "sources": []}


    def tool_node(state: CapstoneState):
        text = state.get("document_text", "")
        risk = 0
        issues = []

        t = text.lower()

        if "unlimited liability" in t:
            risk += 0.4
            issues.append("Unlimited liability detected")

        if "terminate" not in t:
            risk += 0.3
            issues.append("Missing termination clause")

        if "confidential" not in t:
            risk += 0.3
            issues.append("Missing confidentiality clause")

        return {"tool_result": str({"risk_score": round(risk, 2), "issues": issues})}


    def answer_node(state: CapstoneState):
        context = ""
        if state.get("retrieved"):
            context += f"KNOWLEDGE BASE:{state['retrieved']}"
        if state.get("tool_result"):
            context += f"TOOL RESULT:{state['tool_result']}"
        prompt = f"""You are an AI Legal Contract Risk Analyzer.

    Rules:
    - Use ONLY the given context
    - Do NOT hallucinate
    - If unsure → say "I don't have that information"

    Answer in this format:
    1. Summary
    2. Issues
    3. Risk
    4. Verdict

    {context}

    Question: {state['question']}
    """
        response = llm.invoke(prompt)
        return {"answer": response.content}


    def eval_node(state):
        faith = state.get("faithfulness", 1.0)

        if faith < 0.7 and state["eval_retries"] < 2:
            return {
            "eval_retries": state["eval_retries"] + 1,
            "route": "answer"
        }

        return {
            "route": "save"
    }
    def tool_node(state):
        q = state["question"].lower()

        if "date" in q:
            return {"tool_result": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}"}

        if "calculate" in q:
            try:
                expr = q.replace("calculate", "")
                return {"tool_result": str(eval(expr))}
            except:
                return {"tool_result": "Invalid calculation"}

        return {"tool_result": "No tool used"}

    def save_node(state: CapstoneState):
        messages = state.get("messages", [])
        messages.append({"role": "assistant", "content": state["answer"]})
        return {"messages": messages}


    # ===== DECISIONS =====

    def route_decision(state):
        return state["route"]


    def eval_decision(state):
        if state["faithfulness"] < 0.7 and state["eval_retries"] < 2:
            return "answer"
        return "save"


    # ===== GRAPH =====

    graph = StateGraph(CapstoneState)

    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory", "router")

    graph.add_conditional_edges("router", route_decision, {
        "retrieve": "retrieve",
        "tool": "tool",
        "memory_only": "skip"
    })

    graph.add_edge("retrieve", "answer")
    graph.add_edge("tool", "answer")
    graph.add_edge("skip", "answer")

    graph.add_edge("answer", "eval")

    graph.add_conditional_edges("eval", eval_decision, {
        "answer": "answer",
        "save": "save"
    })

    graph.add_edge("save", END)

    agent_app = graph.compile(checkpointer=MemorySaver())

    return agent_app, embedder, collection


try:
    agent_app, embedder, collection = load_agent()
    st.success(f"✅ Knowledge base loaded — {collection.count()} documents")
except Exception as e:
    import traceback
    st.error(traceback.format_exc())
    st.stop()

# ── Session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write(f"{DOMAIN_DESCRIPTION}")
    st.write(f"Session: {st.session_state.thread_id}")
    st.divider()
    st.write("**Topics covered:**")
    KB_TOPICS = [d["topic"] for d in DOCUMENTS]
    for t in KB_TOPICS:
        st.write(f"• {t}")
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# ── Display history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---- Chat memory ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Display old messages ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---- Input ----
if prompt := st.chat_input("Ask something..."):

    # Save + show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # ---- Assistant ----
    with st.chat_message("assistant"):
        with st.spinner("Analyzing contract..."):

            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            result = agent_app.invoke(
                {
                    "question": prompt,
                    "document_text": "This contract has unlimited liability and no termination clause"
                },
                config=config
            )

            answer = result.get("answer", "Sorry, I could not generate an answer.")
            st.write(answer)

            answer_lower = answer.lower()

            # 🔥 Risk badge
            answer_lower = answer.lower()
            if "high" in answer_lower:
                st.error("🚨 High Risk")
            elif "medium" in answer_lower:
                st.warning("⚠️ Medium Risk")
            else:
                st.success("✅ Low Risk")

            # 📊 Faithfulness + sources (your existing feature)
            faith = result.get("faithfulness", 0.0)
            if faith > 0:
                st.caption(f"Faithfulness: {faith:.2f} | Sources: {result.get('sources', [])}")

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})