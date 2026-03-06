---
name: invisible-exec
description: Solve general-purpose tasks by writing and running throwaway code instead of reasoning through them manually
triggers:
  - "calculate"
  - "analyze data"
  - "convert"
  - "parse"
  - "extract"
  - "transform"
  - "count"
  - "compare files"
  - "find duplicates"
  - "sort"
  - "filter"
  - "validate"
  - "check"
---

# Invisible Code Execution

When the user asks you to perform a task that's easier to solve with code than with reasoning alone, write a quick Python script and execute it. The user cares about the **result**, not the code.

## The Pattern

1. Write a Python script to a temp file (`/tmp/ag3nt_scratch_<hash>.py` or similar)
2. Execute it with `exec_command`
3. Return only the **result** to the user (not the code, unless they ask)
4. Clean up the temp file

## When to Use

- Data transformation or calculation ("convert this CSV to JSON")
- Numeric computation ("what's the compound interest on...")
- File analysis ("how many lines in each file matching...")
- Complex text parsing/extraction ("extract all emails from this file")
- Format conversion ("convert this XML to YAML")
- Data validation ("check if this JSON is valid")
- Sorting, filtering, deduplication of data
- Any task where code is more reliable than mental math or reasoning

## When NOT to Use

- Simple questions that don't need computation ("what is 2+2")
- Tasks where the user explicitly wants to see the code ("write a script that...")
- Tasks that require persistent code (the script is throwaway)
- Tasks involving sensitive data that shouldn't be written to disk

## Example

**User:** "How many unique IP addresses are in access.log?"

**You (internal):** Write and run:
```python
ips = set()
with open("access.log") as f:
    for line in f:
        parts = line.split()
        if parts:
            ips.add(parts[0])
print(len(ips))
```

**You (to user):** "There are 1,247 unique IP addresses in access.log."

## Guidelines

- **Speed over elegance** — this is throwaway code, don't over-engineer
- **Print the answer** — the script's stdout is what you'll relay to the user
- **Handle errors** — wrap in try/except and give a clear error if something fails
- **Clean up** — always remove the temp file after execution
- **Be invisible** — the user should feel like you just *know* the answer
