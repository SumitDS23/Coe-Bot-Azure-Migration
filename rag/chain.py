"""
rag/chain.py
Builds the RAG chain: retriever -> prompt -> LLM -> response.
Modified to use Azure OpenAI instead of Gemini/Gemma/Vertex AI.
"""

import logging
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are COE-chatbot, a helpful assistant for Analytics at ABC.
Answer questions using the provided document context.
If the context doesn't contain enough information, say so honestly.
Be concise and precise."""


def build_llm():
    logger.info(f"Using Azure OpenAI LLM: {settings.azure_openai_deployment_name}")
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment_name,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=0,
    )


def build_chain(retriever):
    llm = build_llm()

    # Azure OpenAI (GPT-4o) supports system messages natively
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

    # Convert history tuples to LangChain messages
    lc_history = []
    for human, ai in history:
        lc_history.append(HumanMessage(content=human))
        lc_history.append(AIMessage(content=ai))

    # Retrieve relevant docs
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Run chain
    response = chain.invoke({
        "question": question,
        "context": context,
        "history": lc_history,
    })

    return response.strip(), docs