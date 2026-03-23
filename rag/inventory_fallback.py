"""
rag/inventory.py
Dynamic inventory query engine with:
- DuckDB querying directly on DataFrame
- Follow-up question handling via accumulated_filters
- Topic shift detection with user confirmation
- Natural language responses via Gemini
"""

import json
import logging
import re
import pandas as pd
import duckdb
from pathlib import Path
from config.settings import settings

logger = logging.getLogger(__name__)

_df: pd.DataFrame = None

SCHEMA_INFO = """
You are querying a DataFrame called `df` with these columns:

| Column               | Known Values                                                        |
|----------------------|---------------------------------------------------------------------|
| LOB                  | Life Insurance, Health Insurance, Housing Finance, Mutual Funds,    |
|                      | Bajaj Digital, Central Analytics Unit, Finance Consumer, Finance Risk|
| Function             | Cross-Sell, HR, Pre-Sales, Revenue Retention, Risk Management,      |
|                      | Up-Sell, Fraud, Engagement, Claims, Underwriting, etc.              |
| Model_Name           | (free text)                                                         |
| Model_Description    | (free text)                                                         |
| ML_Non_ML            | ML, Non-ML                                                          |
| Status               | Live, WIP                                                           |
| Timeline             | FY25, FY26 End, etc.                                                |
| Owner                | (free text)                                                         |
| Document_Availability| Yes, No                                                             |

LOB Abbreviations:
- SLI / ABSLI / Sun Life          -> Life Insurance
- HI / ABHI                       -> Health Insurance
- HFL / HFC                       -> Housing Finance
- AMC / ABSLAMC                   -> Mutual Funds
- CAU                             -> Central Analytics Unit
- FL CA                           -> Finance Consumer
- FL RA                           -> Finance Risk

Status aliases: live -> Live | wip/in progress -> WIP
ML aliases: ml/machine learning -> ML | non ml/non-ml -> Non-ML
"""

FOLLOWUP_SIGNALS = [
    "of these", "of them", "out of these", "out of them",
    "how many of", "among these", "among them", "within these",
    "from these", "from them", "these models", "those models",
    "in them", "further", "also", "and how many", "what about",
    "which of", "do any of", "are any of"
]

EXACT_MATCH_COLS = {"ML_Non_ML", "Status", "Document_Availability"}


def load_inventory() -> pd.DataFrame:
    global _df
    if _df is not None:
        return _df
    path = settings.inventory_path
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Inventory file not found at '{path}'. Set INVENTORY_PATH in your .env file."
        )
    df = pd.read_excel(path)
    df.columns = [c.strip().replace(" ", "_").replace("/", "_") for c in df.columns]
    _df = df
    logger.info(f"Inventory loaded: {len(df)} rows from {path}")
    return df


def get_inventory_stats() -> dict:
    try:
        df = load_inventory()
        return {
            "total": len(df),
            "live": len(df[df["Status"] == "Live"]),
            "wip": len(df[df["Status"] == "WIP"]),
            "lobs": df["LOB"].nunique(),
        }
    except Exception as e:
        logger.warning(f"Could not load inventory stats: {e}")
        return {"total": "N/A", "live": "N/A", "wip": "N/A", "lobs": "N/A"}


def parse_question(question: str, accumulated_filters: dict, turn_history: list) -> dict:
    """Use Gemini to parse question into structured filters and detect follow-ups."""
    from google import genai
    client = genai.Client(api_key=settings.google_api_key_llm)

    history_text = "\n".join(
        f"Turn {i+1}: Q='{h['question']}' filters={h['filters']} results={h['result_count']}"
        for i, h in enumerate(turn_history[-4:])
    ) if turn_history else "None - first question."

    signals = [s for s in FOLLOWUP_SIGNALS if s in question.lower()]
    signal_hint = f"Follow-up signal words detected: {signals}" if signals else ""

    prompt = f"""
{SCHEMA_INFO}

CONVERSATION HISTORY (last 4 turns):
{history_text}

ACCUMULATED FILTERS (from prior follow-ups):
{json.dumps(accumulated_filters) if accumulated_filters else "{}"}

NEW USER QUESTION: "{question}"
{signal_hint}

Return ONLY a JSON object. No markdown, no explanation.

Rules:
1. is_followup: TRUE if question references prior results using words like
   "these", "them", "those", "of these", etc. OR clearly narrows a prior result set.
   FALSE if completely new independent topic.

2. topic_shift: TRUE only if is_followup=false AND accumulated_filters is not empty
   AND the new question targets a DIFFERENT LOB or main entity than before.

3. new_filters: only filters explicitly in THIS question. Omit keys not mentioned.

4. final_filters:
   - is_followup=true  -> MERGE accumulated_filters + new_filters
   - is_followup=false -> new_filters only (fresh start)

5. query_type:
   - "count"     -> how many, count, total
   - "list"      -> list, show, which models
   - "describe"  -> details about a specific named model
   - "breakdown" -> split/group by a column

6. group_by: column name if breakdown, else null.

Return exactly:
{{
  "is_followup": <true|false>,
  "topic_shift": <true|false>,
  "new_filters": {{}},
  "final_filters": {{}},
  "query_type": "<count|list|describe|breakdown>",
  "group_by": "<column or null>",
  "explanation": "<one line: what filters are active and why>"
}}
"""

    try:
        response = client.models.generate_content(
            model=settings.llm_model,
            config={"temperature": 0},
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        text = re.sub(r"```(?:json)?", "", response.text.strip()).strip().rstrip("```").strip()
        return json.loads(text)
    except Exception as e:
        logger.error(f"parse_question failed: {e}")
        return {
            "is_followup": False, "topic_shift": False,
            "new_filters": {}, "final_filters": {},
            "query_type": "list", "group_by": None,
            "explanation": f"Parse error: {e}",
        }


def build_where(filters: dict) -> str:
    conditions = []
    for col, val in filters.items():
        if not val:
            continue
        safe_val = val.replace("'", "''")
        if col in EXACT_MATCH_COLS:
            conditions.append(f"LOWER({col}) = '{safe_val.lower()}'")
        else:
            conditions.append(f"LOWER({col}) LIKE '%{safe_val.lower()}%'")
    return ("WHERE " + " AND ".join(conditions)) if conditions else ""


def run_duck_query(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    return duckdb.query(sql).to_df()


def generate_nl_response(question: str, query_type: str, result_df: pd.DataFrame,
                          turn_history: list, final_filters: dict) -> str:
    """Pass raw results to Gemini for a natural language answer."""
    from google import genai
    client = genai.Client(api_key=settings.google_api_key_llm)

    history_text = "\n".join(f"Q: {h['question']}" for h in turn_history[-4:]) if turn_history else "First question."

    if result_df.empty:
        result_summary = "No results found."
    elif query_type == "count":
        result_summary = f"Count: {result_df.iloc[0, 0]}"
    elif query_type == "breakdown":
        result_summary = result_df.to_string(index=False)
    else:
        preview_cols = [c for c in ["Model_Name", "LOB", "Function", "Status",
                                     "Owner", "Timeline", "Model_Description"]
                        if c in result_df.columns]
        result_summary = result_df[preview_cols].to_string(index=False)

    prompt = f"""
You are Converge Knowledge, a friendly and professional COE Analytics assistant.

CONVERSATION SO FAR:
{history_text}

USER QUESTION: "{question}"
ACTIVE FILTERS: {json.dumps(final_filters) if final_filters else "none"}
QUERY TYPE: {query_type}

DATABASE RESULTS:
{result_summary}

Respond in natural, conversational language. Guidelines:
- For counts: state the number clearly with context.
  e.g. "There are 12 live models in Life Insurance's Cross-Sell function."
- For lists <= 5 results: mention all model names in your response naturally.
- For lists > 5 results: give a brief summary — total count and key patterns.
  End with "Here's the full list below." so the user knows a table follows.
- For breakdowns: highlight the most interesting insight from the distribution.
- For describes: give a clear, readable summary of the model's details.
- For empty results: explain what was searched and suggest trying broader criteria.
- Never mention SQL, DuckDB, DataFrames, or any technical internals.
- Never say "based on the data" or "according to the results".
- Keep it concise — 2 to 4 sentences max unless it is a describe query.
"""

    try:
        response = client.models.generate_content(
            model=settings.llm_model,
            config={"temperature": 0.3},
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"NL response failed: {e}")
        if result_df.empty:
            return "No models found matching your criteria. Try broadening your search."
        if query_type == "count":
            return f"Found **{result_df.iloc[0, 0]}** model(s) matching your criteria."
        return f"Found **{len(result_df)}** model(s) matching your criteria."


def run_inventory_query(question: str, accumulated_filters: dict,
                        turn_history: list) -> tuple:
    """
    Full pipeline: parse -> query -> natural language response.

    Returns:
        (nl_answer, result_df_or_None, updated_filters, topic_shift, parsed)
    """
    try:
        df = load_inventory()
    except FileNotFoundError as e:
        return str(e), None, accumulated_filters, False, {}

    # Step 1 - Parse intent and filters
    parsed = parse_question(question, accumulated_filters, turn_history)
    final_filters = parsed.get("final_filters", {})
    query_type    = parsed.get("query_type", "list")
    group_by      = parsed.get("group_by")
    topic_shift   = parsed.get("topic_shift", False)

    # Step 2 - Build and run SQL
    where = build_where(final_filters)

    if query_type == "count":
        sql = f"SELECT COUNT(*) AS count FROM df {where}"
    elif query_type == "breakdown":
        col = group_by or "Function"
        sql = f"SELECT {col}, COUNT(*) AS model_count FROM df {where} GROUP BY {col} ORDER BY model_count DESC"
    elif query_type == "describe":
        sql = f"SELECT * FROM df {where} LIMIT 1"
    else:
        sql = f"SELECT Model_Name, LOB, Function, Status, ML_Non_ML, Owner, Timeline FROM df {where}"

    try:
        result_df = run_duck_query(sql, df)
    except Exception as e:
        logger.error(f"DuckDB failed: {e} | SQL: {sql}")
        return (
            "I had trouble retrieving that information. Could you rephrase your question?",
            None, accumulated_filters, False, parsed
        )

    # Step 3 - Natural language response
    nl_answer = generate_nl_response(
        question=question,
        query_type=query_type,
        result_df=result_df,
        turn_history=turn_history,
        final_filters=final_filters,
    )

    # Only return dataframe for list/breakdown with more than 5 rows
    show_df = None
    if query_type in ("list", "breakdown") and len(result_df) > 5:
        show_df = result_df

    return nl_answer, show_df, final_filters, topic_shift, parsed
