
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
    
    # ---------------------------------

    # messages = [
    #     {
    #         "role": "system",
    #         "content": SYSTEM_PROMPT
    #     },
    #     {
    #         "role": "user",
    #         "content": user_request
    #     }
    # ]

    # print("\nAgent is working...\n")
    # ------------------------------------------------

    # Run mandatory checks using Python.
    verified_results = collect_verified_results()

    # Format verified results for the LLM.
    verified_context = json.dumps(
        verified_results,
        ensure_ascii=False,
        indent=2
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
            + """

IMPORTANT DATA ACCURACY RULES:

The user message contains verified results
calculated directly by Python.

Use those results as the authoritative
source of all numerical findings.

Never invent missing-value counts,
duplicate counts, or sample records.

If a result is unavailable, clearly state
that it was not available.
"""
        },
        {
            "role": "user",
            "content": (
                user_request
                + "\n\nVERIFIED PYTHON RESULTS:\n"
                + verified_context
            )
        }
    ]

    print("\nAgent is working...\n")

    # Keep the existing agent execution loop below.    

    for step in range(1, MAX_TOOL_ROUNDS + 1):

        # Send conversation and available tools to LLM.
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            options={"temperature": 0}
        )

        assistant_message = response.message

        # Preserve the assistant response in conversation history.
        messages.append(assistant_message)

        tool_calls = assistant_message.tool_calls or []

        # No more tools requested: agent has responded.
        if not tool_calls:

            print("\n========== AGENT RESPONSE ==========\n")
            print(assistant_message.content)
            print("\n====================================\n")

            return

        # Execute each requested tool.
        for tool_call in tool_calls:

            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print(f"[Agent] Calling tool: {tool_name}")
            print(f"[Agent] Arguments: {arguments}")

            function = AVAILABLE_TOOLS.get(tool_name)

            if function is None:

                result = {
                    "error": f"Unknown tool: {tool_name}"
                }

            else:

                try:

                    result = function(**arguments)

                except Exception as error:

                    result = {
                        "error": str(error)
                    }

            print(
                "[Python Tool Result]",
                json.dumps(result, ensure_ascii=False)
            )

            # Send the tool result back to the LLM.
            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(
                    result,
                    ensure_ascii=False
                )
            })

    print(
        "\nAgent stopped: maximum tool-call rounds reached."
    )


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
