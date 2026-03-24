"""
rag/router.py
Intents: INVENTORY or RAG only.
Modified to use Azure OpenAI instead of Gemini/Gemma.
"""
import logging
import json
import re
from enum import Enum
from pydantic import BaseModel
from openai import AzureOpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

class Intent(str, Enum):
    INVENTORY = "INVENTORY"
    RAG       = "RAG"

class RouterDecision(BaseModel):
    intent: Intent
    reasoning: str

ROUTER_PROMPT = """
You are an intent classifier for a COE Analytics chatbot with TWO data sources:

1. INVENTORY (Excel) - Enterprise Model Inventory containing model names,
   LOBs, Functions, Status, Owners, Timelines, Documentation availability.

2. RAG (Vector Store) - Internal documents about analytics methodologies,
   model processes, documentation content, company policies.

ALWAYS ROUTE TO INVENTORY for ANY question that involves:
- Counting models: "how many", "count", "total", "number of"
- Listing models: "list", "show", "which models", "what models"
- Model status: "live", "WIP", "in progress"
- Model ownership: "who owns", "owned by"
- Filtering by LOB: SLI, AMC, HI, HFC, Life Insurance, Mutual Funds, etc.
- Filtering by Function: Cross-Sell, HR, Fraud, etc.
- Model details from the inventory: timeline, documentation availability

ONLY ROUTE TO RAG for questions about:
- How a model works technically
- Methodology or approach used
- Process explanations
- Policy or guideline content
- Concepts that require reading a document

CRITICAL EXAMPLES:
"How many live models are there in SLI?"         -> INVENTORY
"How many models does Life Insurance have?"       -> INVENTORY
"List all live SLI models"                        -> INVENTORY
"How many WIP models are in Cross-Sell?"          -> INVENTORY
"Which AMC models have documentation?"            -> INVENTORY
"Who owns the fraud detection model?"             -> INVENTORY
"How many models are live?"                       -> INVENTORY
"How does the persistency model work?"            -> RAG
"What methodology is used for fraud detection?"   -> RAG
"Explain the churn prediction approach"           -> RAG

Return valid JSON only in this format:
{"intent": "INVENTORY" or "RAG", "reasoning": "your reasoning here"}
"""


def classify_intent(question: str, last_intent: str = "") -> RouterDecision:
    # Build context hint based on last intent
    context_hint = ""
    if last_intent == "INVENTORY":
        context_hint = (
            "\nCONTEXT: The previous question was answered from the INVENTORY. "
            "If this question is a follow-up or continuation (e.g. 'list all', "
            "'show me', 'which ones', 'can you list'), route to INVENTORY."
        )
    elif last_intent == "RAG":
        context_hint = (
            "\nCONTEXT: The previous question was answered from the RAG documents. "
            "If this question is a follow-up about the same topic, route to RAG."
        )

    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )

    # Build messages
    messages = [
        {
            "role": "system",
            "content": ROUTER_PROMPT + context_hint
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # Call Azure OpenAI
    response = client.chat.completions.create(
        model=settings.azure_openai_deployment_name,  # e.g. gpt-4o
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},      # enforces JSON output
    )

    # Parse response
    text = response.choices[0].message.content.strip()
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()

    try:
        data = json.loads(text)
        decision = RouterDecision(
            intent=Intent(data["intent"].upper()),
            reasoning=data.get("reasoning", ""),
        )
        logger.info(f"Routed to: {decision.intent} | Reason: {decision.reasoning}")
        return decision
    except Exception as e:
        logger.error(f"Router parse failed: {e} | raw: {text}")
        return RouterDecision(intent=Intent.RAG, reasoning="parse fallback")