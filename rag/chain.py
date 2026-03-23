"""
rag/chain.py
Builds the RAG chain: retriever -> prompt -> LLM -> response.
Gemma-compatible: system prompt inlined when model doesn't support system messages.
"""

import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are COE-chatbot, a helpful assistant for Analytics at ABC.
Answer questions using the provided document context.
If the context doesn't contain enough information, say so honestly.
Be concise and precise."""


def _is_gemma(model_name: str) -> bool:
    return "gemma" in model_name.lower()


def build_llm():
    use_vertex = os.environ.get("USE_VERTEX", "false").lower() == "true"
    if use_vertex:
        from langchain_google_vertexai import ChatVertexAI
        logger.info(f"Using Vertex AI LLM: {settings.llm_model}")
        return ChatVertexAI(
            model_name=settings.llm_model,
            temperature=0,
            project=settings.gcp_project,
            location=settings.gcp_region,
        )
    else:
        logger.info(f"Using Gemini API LLM: {settings.llm_model}")
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=0,
            google_api_key=settings.google_api_key_llm,
        )


def build_chain(retriever):
    llm = build_llm()

    if _is_gemma(settings.llm_model):
        # Gemma: no system message supported — inline into human message
        logger.info("Building Gemma-compatible RAG chain (no system message)")
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            ("human", (
                f"{SYSTEM_PROMPT}\n\n"
                "Context from company documents:\n\n{context}\n\n"
                "---\n\n{question}"
            )),
        ])
    else:
        # Gemini and others: standard system + human structure
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Context from company documents:\n\n{context}\n\n---\n\n{question}"),
        ])

    chain = prompt | llm | StrOutputParser()
    logger.info("RAG chain built successfully")
    return chain


def run_rag(chain, retriever, question: str, history: list) -> tuple[str, list]:
    from langchain_core.messages import HumanMessage, AIMessage

    lc_history = []
    for human, ai in history:
        lc_history.append(HumanMessage(content=human))
        lc_history.append(AIMessage(content=ai))

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    response = chain.invoke({
        "question": question,
        "context": context,
        "history": lc_history,
    })

    return response.strip(), docs