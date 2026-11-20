
import csv
import json
from pathlib import Path

import ollama


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL = "llama3.2"  # Change this to your installed model if necessary.

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "customers.csv"

MAX_TOOL_ROUNDS = 6


# ==========================================
# 2. DATA ACCESS
# ==========================================

def load_dataset():
    """
    Load the configured CSV dataset.

    The agent can only access this predefined file.
    """

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}"
        )

    with CSV_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV file has no header.")

        columns = reader.fieldnames
        rows = list(reader)

    return columns, rows

# ==========================================
def read_multiline_request():

    print("\nEnter your request.")
    print("Type END on a separate line when finished.\n")

    lines = []

    while True:

        line = input()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    return "\n".join(lines).strip()

# ==========================================
# 3. DEFINE PYTHON TOOLS
# ==========================================

def get_dataset_overview() -> dict:
    """
    Get the dataset's basic structure.

    Returns:
        Number of rows, columns and column names.
    """

    columns, rows = load_dataset()

    return {
        "file_name": CSV_PATH.name,
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": columns
    }


def get_null_counts() -> dict:
    """
    Count missing or empty values in each column.

    Empty strings and whitespace-only values
    are treated as missing values.
    """

    columns, rows = load_dataset()

    null_counts = {}

    for column in columns:

        count = sum(
            1
            for row in rows
            if not (row.get(column) or "").strip()
        )

        null_counts[column] = count

    return {
        "missing_values": null_counts
    }


def get_duplicate_row_count() -> dict:
    """
    Count exact duplicate rows across all columns.

    Each repeated occurrence after the first
    is counted as a duplicate.
    """

    columns, rows = load_dataset()

    seen_rows = set()
    duplicate_count = 0

    for row in rows:

        row_key = tuple(
            row.get(column) or ""
            for column in columns
        )

        if row_key in seen_rows:
            duplicate_count += 1
        else:
            seen_rows.add(row_key)

    return {
        "duplicate_rows": duplicate_count
    }


def get_sample_rows(limit: int = 5) -> dict:
    """
    Return sample records from the dataset.

    Args:
        limit: Number of rows to return (maximum 10).
    """

    columns, rows = load_dataset()

    limit = max(1, min(int(limit), 10))

    return {
        "columns": columns,
        "returned_rows": min(limit, len(rows)),
        "rows": rows[:limit]
    }


# ==========================================
# 4. REGISTER AVAILABLE TOOLS
# ==========================================

AVAILABLE_TOOLS = {
    "get_dataset_overview": get_dataset_overview,
    "get_null_counts": get_null_counts,
    "get_duplicate_row_count": get_duplicate_row_count,
    "get_sample_rows": get_sample_rows
}

# The Ollama Python SDK converts these
# Python functions into tool definitions.

TOOLS = list(AVAILABLE_TOOLS.values())


# ==========================================
# 5. AGENT INSTRUCTIONS
# ==========================================

SYSTEM_PROMPT = """
You are an AI Data Quality Assistant.

Your responsibility is to inspect datasets
and provide accurate data quality reports.

You have access to these tools:

1. get_dataset_overview
2. get_null_counts
3. get_duplicate_row_count
4. get_sample_rows

Rules:

- Use tools to obtain actual dataset information.
- Never invent row counts or data quality results.
- For a complete data quality report, inspect
  the overview, missing values and duplicates.
- Use sample rows when additional inspection
  is useful.
- Explain the findings clearly.
- Suggest appropriate data quality improvements.
- Do not claim that data has been modified.
- All available tools are read-only.

You are an analytical assistant.
Be precise and factual.
"""

# ==== Prior Verification ====================
def collect_verified_results():

    print("\nRunning mandatory data quality checks...\n")

    results = {
        "dataset_overview": get_dataset_overview(),
        "missing_values": get_null_counts(),
        "duplicate_rows": get_duplicate_row_count(),
        "sample_rows": get_sample_rows()
    }

    print("Mandatory checks completed.")

    return results
# ==========================================

# ==========================================
# 6. AGENT EXECUTION LOOP
# ==========================================


def run_agent(user_request: str):

    # STEP 1: Execute Python functions.
    verified_results = collect_verified_results()

    # STEP 2: Prepare the LLM conversation.
    messages = [
        {
            "role": "system",
            "content": """
You are an AI Data Quality Report Analyst.

Your job is to analyze verified Python results
and generate a professional data quality report.

IMPORTANT RULES:

1. The supplied JSON contains actual results
   calculated directly by Python.

2. Use these results as your only source
   for numerical findings.

3. Never invent missing values, duplicate
   counts, row counts or sample records.

4. Do not call tools.

5. Do not generate tool-call JSON.

6. Explain identified issues and suggest
   appropriate data quality improvements.

7. Generate the final response in Markdown.
"""
        },
        {
            "role": "user",
            "content": (
                f"USER REQUEST:\n{user_request}\n\n"
                "VERIFIED PYTHON RESULTS:\n"
                + json.dumps(
                    verified_results,
                    ensure_ascii=False,
                    indent=2
                )
            )
        }
    ]

    # STEP 3: Ask LLM to analyze verified results.
    print("[LLM] Analyzing verified results...\n")

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={"temperature": 0}
    )

    # STEP 4: Display final response.
    final_answer = response.message.content

    print("\n========== FINAL DATA QUALITY REPORT ==========\n")

    if final_answer:
        print(final_answer)
    else:
        print("The model returned an empty response.")

    print("\n================================================\n")



# ==========================================
# 7. APPLICATION ENTRY POINT
# ==========================================

# if __name__ == "__main__":

#     print("====================================")
#     print("       AI DATA QUALITY AGENT        ")
#     print("====================================")

#     user_request = input("\nYou: ").strip()

#     if not user_request:
#         print("Please enter a request.")
#     else:
#         run_agent(user_request)

if __name__ == "__main__":

    print("====================================")
    print("       AI DATA QUALITY AGENT        ")
    print("====================================")

    user_request = read_multiline_request()

    if not user_request:
        print("Please enter a request.")
    else:
        run_agent(user_request)
