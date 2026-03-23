"""
rag/router.py
Intents: INVENTORY or RAG only.
"""
import logging
import json
import re
from enum import Enum
from pydantic import BaseModel
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

Return valid JSON only.
"""
## Gemini Model Function (use for Gemini models - supports system_instruction and structured output)
# def classify_intent(question: str, last_intent: str = "") -> RouterDecision:
#     from google import genai
#     client = genai.Client(api_key=settings.google_api_key_llm)
#
#     context_hint = ""
#     if last_intent == "INVENTORY":
#         context_hint = (
#             "\nCONTEXT: The previous question was answered from the INVENTORY. "
#             "If this question is a follow-up or continuation (e.g. 'list all', "
#             "'show me', 'which ones', 'can you list'), route to INVENTORY."
#         )
#     elif last_intent == "RAG":
#         context_hint = (
#             "\nCONTEXT: The previous question was answered from the RAG documents. "
#             "If this question is a follow-up about the same topic, route to RAG."
#         )
#
#     system_prompt_with_context = ROUTER_PROMPT + context_hint
#
#     response = client.models.generate_content(
#         model=settings.llm_model,
#         config={
#             "system_instruction": system_prompt_with_context,
#             "response_mime_type": "application/json",
#             "response_schema": RouterDecision,
#             "temperature": 0,
#         },
#         contents=[{"role": "user", "parts": [{"text": question}]}],
#     )
#     decision = response.parsed
#     logger.info(f"Routed to: {decision.intent} | Reason: {decision.reasoning}")
#     return decision
##Gemma model function
def classify_intent(question: str, last_intent: str = "") -> RouterDecision:
    from google import genai
    client = genai.Client(api_key=settings.google_api_key_llm)

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

    full_prompt = f"{ROUTER_PROMPT}{context_hint}\n\nUser question: {question}"

    response = client.models.generate_content(
        model=settings.llm_model,
        config={"temperature": 0},
        contents=[{"role": "user", "parts": [{"text": full_prompt}]}],
    )

    text = response.text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()

    try:
        data = json.loads(text)
        return RouterDecision(
            intent=Intent(data["intent"].upper()),
            reasoning=data.get("reasoning", ""),
        )
    except Exception as e:
        logger.error(f"Router parse failed: {e} | raw: {text}")
        return RouterDecision(intent=Intent.RAG, reasoning="parse fallback")