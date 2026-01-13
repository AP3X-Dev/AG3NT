# Tools To Add - Implementation Status

This document tracks tools from external sources that have been considered for DeepAgents.

## Implementation Status

| Tool | Status | Notes |
|------|--------|-------|
| Bash | ✅ MERGED | Improvements merged into `execute` tool in FilesystemMiddleware |
| create_file | ✅ MERGED | Merged into `write_file` with overwrite capability |
| edit_file | ✅ UPDATED | Enhanced description in FilesystemMiddleware |
| find_thread | ❌ SKIPPED | AG3NT-specific, requires thread infrastructure |
| finder | ✅ IMPLEMENTED | In AdvancedMiddleware |
| format_file | ✅ IMPLEMENTED | In UtilitiesMiddleware |
| get_diagnostics | ✅ IMPLEMENTED | In UtilitiesMiddleware |
| glob | ✅ UPDATED | Enhanced description in FilesystemMiddleware |
| Grep | ✅ UPDATED | Enhanced description in FilesystemMiddleware |
| librarian | ✅ IMPLEMENTED | As subagent specification |
| look_at | ✅ IMPLEMENTED | In AdvancedMiddleware |
| mermaid | ✅ IMPLEMENTED | In UtilitiesMiddleware |
| oracle | ✅ IMPLEMENTED | As subagent specification |
| Read | ✅ UPDATED | Enhanced description in FilesystemMiddleware |
| read_mcp_resource | ❌ SKIPPED | Requires MCP server infrastructure |
| read_thread | ❌ SKIPPED | AG3NT-specific, requires thread infrastructure |
| read_web_page | ✅ IMPLEMENTED | In WebMiddleware |
| skill | ✅ EXISTS | SkillsMiddleware provides this functionality |
| undo_edit | ✅ IMPLEMENTED | In UtilitiesMiddleware |
| web_search | ✅ IMPLEMENTED | In WebMiddleware |

---

## Original Tool Specifications (Reference)

### Bash
Executes the given shell command using bash (or sh on systems without bash)

Executes the given shell command using bash (or sh on systems without bash). - Do NOT chain commands with `;` or `&&` or use `&` for background processes; make separate tool calls instead - Do NOT use interactive commands (REPLs, editors, password prompts) - Output is truncated to the last 50000 characters - Environment variables and `cd` do not persist between commands; use the `cwd` parameter instead - Commands run in the workspace root by default; only use `cwd` when you need a different directory (never use `cd dir && cmd`) - Only the last 50000 characters of the output will be returned to you along with how many lines got truncated, if any; rerun with a grep or head/tail filter if needed - On Windows, use PowerShell commands and `\` path separators - ALWAYS quote file paths: `cat "path with spaces/file.txt"` - Use finder/Grep instead of find/grep, Read instead of cat, edit_file instead of sed - Only run `git commit` and `git push` if explicitly instructed by the user.




create_file
Create or overwrite a file in the workspace

Create or overwrite a file in the workspace. Use this tool when you want to create a new file with the given content, or when you want to replace the contents of an existing file. Prefer this tool over `edit_file` when you want to ovewrite the entire contents of a file.



edit_file
Make edits to a text file

Make edits to a text file. Replaces `old_str` with `new_str` in the given file. Returns a git-style diff showing the changes made as formatted markdown, along with the line range ([startLine, endLine]) of the changed content. The diff is also shown to the user. The file specified by `path` MUST exist, and it MUST be an absolute path. If you need to create a new file, use `create_file` instead. `old_str` MUST exist in the file. Use tools like `Read` to understand the files you are editing before changing them. `old_str` and `new_str` MUST be different from each other. Set `replace_all` to true to replace all occurrences of `old_str` in the file. Else, `old_str` MUST be unique within the file or the edit will fail. Additional lines of context can be added to make the string more unique. If you need to replace the entire contents of a file, use `create_file` instead, since it requires less tokens for the same action (since you won't have to repeat the contents before replacing)



find_thread
Find AG3NT threads (conversation threads with the agent) using a query DSL

Find AG3NT threads (conversation threads with the agent) using a query DSL. ## What this tool finds This tool searches **AG3NT threads** (conversations with the agent), NOT git commits. Use this when the user asks about threads, conversations, or AG3NT history. ## Query syntax - **Keywords**: Bare words or quoted phrases for text search: `auth` or `"race condition"` - **File filter**: `file:path` to find threads that touched a file: `file:src/auth/login.ts` - **Repo filter**: `repo:url` to scope to a repository: `repo:github.com/owner/repo` or `repo:owner/repo` - **Author filter**: `author:name` to find threads by a user: `author:alice` or `author:me` for your own threads - **Date filters**: `after:date` and `before:date` to filter by date: `after:2024-01-15`, `after:7d`, `before:2w` - **Task filter**: `task:id` to find threads that worked on a task: `task:142`. Use `task:142+` to include threads that worked on the task's dependencies, `task:142^` to include dependents (tasks that depend on this task), or `task:142+^` for both. - **Combine filters**: Use implicit AND: `auth file:src/foo.ts repo:AG3NT after:7d` All matching is case-insensitive. File paths use partial matching. Date formats: ISO dates (`2024-01-15`), relative days (`7d`), or weeks (`2w`). ## When to use this tool - "which thread touched this file" / "which thread modified this file" - "what thread last changed X" / "find the thread that edited X" - "find threads about X" / "search threads mentioning Y" - Any question about AG3NT thread history or previous AG3NT conversations - When the user says "thread" and is referring to AG3NT work, not git commits ## When NOT to use this tool - If the user asks about git commits, git history, or git blame → use git commands instead - If the user wants to know WHO (a person) made changes → use git log # Examples User asks: "Find threads where we discussed the monorepo migration" ```json {"query":"monorepo migration","limit":10} ``` User asks: "Show me threads that modified src/server/index.ts" ```json {"query":"file:src/server/index.ts","limit":5} ``` User asks: "What threads have touched this file?" (for current file in github.com/sourcegraph/AG3NT) ```json {"query":"file:core/src/tools/tool-service.ts repo:sourcegraph/AG3NT"} ``` User asks: "Find auth-related threads in the AG3NT repo" ```json {"query":"auth repo:sourcegraph/AG3NT"} ``` User asks: "Show me my recent threads" ```json {"query":"author:me","limit":10} ``` User asks: "Find threads from the last week about authentication" ```json {"query":"auth after:7d","limit":10} ``` User asks: "Which threads worked on task 142 and its dependencies?" ```json {"query":"task:142+"} ``` User asks: "Show me all threads related to task 50 and tasks that depend on it" ```json {"query":"task:50^"} ```




finder
Intelligently search your codebase: Use it for complex, multi-step search tasks where you need to find code based on functionality or concepts rather than exact matches

Intelligently search your codebase: Use it for complex, multi-step search tasks where you need to find code based on functionality or concepts rather than exact matches. Anytime you want to chain multiple grep calls you should use this tool. WHEN TO USE THIS TOOL: - You must locate code by behavior or concept - You need to run multiple greps in sequence - You must correlate or look for connection between several areas of the codebase. - You must filter broad terms ("config", "logger", "cache") by context. - You need answers to questions such as "Where do we validate JWT authentication headers?" or "Which module handles file-watcher retry logic" WHEN NOT TO USE THIS TOOL: - When you know the exact file path - use Read directly - When looking for specific symbols or exact strings - use glob or Grep - When you need to create, modify files, or run terminal commands USAGE GUIDELINES: 1. Always spawn multiple search agents in parallel to maximise speed. 2. Formulate your query as a precise engineering request. ✓ "Find every place we build an HTTP error response." ✗ "error handling search" 3. Name concrete artifacts, patterns, or APIs to narrow scope (e.g., "Express middleware", "fs.watch debounce"). 4. State explicit success criteria so the agent knows when to stop (e.g., "Return file paths and line numbers for all JWT verification calls"). 5. Never issue vague or exploratory commands - be definitive and goal-oriented.




format_file
Format a file using VS Code's formatter

Get the diagnostics (errors, warnings, etc.) for a file or directory (prefer running for directories rather than files one by one!) Output is shown in the UI so do not repeat/summarize the diagnostics.



get_diagnostics
Get the diagnostics (errors, warnings, etc

Get the diagnostics (errors, warnings, etc.) for a file or directory (prefer running for directories rather than files one by one!) Output is shown in the UI so do not repeat/summarize the diagnostics.



glob
Fast file pattern matching tool that works with any codebase size

Fast file pattern matching tool that works with any codebase size Use this tool to find files by name patterns across your codebase. It returns matching file paths sorted by most recent modification time first. ## File pattern syntax - `**/*.js` - All JavaScript files in any directory - `src/**/*.ts` - All TypeScript files under the src directory (searches only in src) - `*.json` - All JSON files in the current directory - `**/*test*` - All files with "test" in their name - `web/src/**/*` - All files under the web/src directory - `**/*.{js,ts}` - All JavaScript and TypeScript files (alternative patterns) - `src/[a-z]*/*.ts` - TypeScript files in src subdirectories that start with lowercase letters # Examples Find all typescript files in the codebase ```json {"filePattern":"**/*.ts"} ``` Find all test files under a specific directory ```json {"filePattern":"src/**/*test*.ts"} ``` Search for svelte component files in the web/src directory ```json {"filePattern":"web/src/**/*.svelte"} ``` Find the 10 most recently modified JSON files ```json {"filePattern":"**/*.json","limit":10} ```



Grep
Search for exact text patterns in files using ripgrep, a fast keyword search tool

Search for exact text patterns in files using ripgrep, a fast keyword search tool. # When to use this tool - Finding exact text matches (variable names, function calls, specific strings) - Use finder for semantic/conceptual searches # Strategy - Use 'path' or 'glob' to narrow searches; run multiple focused calls rather than one broad search - Uses Rust-style regex (escape `{` and `}`); use `literal: true` for literal text search # Constraints - Results are limited to 100 matches (up to 10 per file) - Lines are truncated at 200 characters # Examples Find a specific function name across the codebase ```json {"pattern":"registerTool","path":"core/src"} ``` Search for interface definitions in a specific directory ```json {"pattern":"interface ToolDefinition","path":"core/src/tools"} ``` Use a case-sensitive search to find the exact string `ERROR:` ```json {"pattern":"ERROR:","caseSensitive":true} ``` Find TODO comments in frontend code ```json {"pattern":"TODO:","path":"web/src"} ``` Find a specific function name in test files ```json {"pattern":"restoreThreads","glob":"**/*.test.ts"} ``` Find all REST API endpoint definitions ```json {"pattern":"app\\.(get|post|put|delete)\\([\"']","path":"server"} ``` Locate CSS class definition in stylesheets ```json {"pattern":"\\.container\\s*\\{","path":"web/src/styles"} ``` # Complementary to finder - Use finder first to locate relevant code concepts - Then use Grep to find specific implementations or all occurrences - For complex tasks, iterate between both tools to refine your understanding



librarian
The Librarian - a specialized codebase understanding agent that helps answer questions about large, complex codebases

The Librarian - a specialized codebase understanding agent that helps answer questions about large, complex codebases. The Librarian works by reading from GitHub - it can see the private repositories the user approved access to in addition to all public repositories on GitHub. The Librarian acts as your personal multi-repository codebase expert, providing thorough analysis and comprehensive explanations across repositories. It's ideal for complex, multi-step analysis tasks where you need to understand code architecture, functionality, and patterns across multiple repositories. WHEN TO USE THE LIBRARIAN: - Understanding complex multi-repository codebases and how they work - Exploring relationships between different repositories - Analyzing architectural patterns across large open-source projects - Finding specific implementations across multiple codebases - Understanding code evolution and commit history - Getting comprehensive explanations of how major features work - Exploring how systems are designed end-to-end across repositories WHEN NOT TO USE THE LIBRARIAN: - Simple local file reading (use Read directly) - Local codebase searches (use finder) - Code modifications or implementations (use other tools) - Questions not related to understanding existing repositories USAGE GUIDELINES: 1. Be specific about what repositories or projects you want to understand 2. Provide context about what you're trying to achieve 3. The Librarian will explore thoroughly across repositories before providing comprehensive answers 4. Expect detailed, documentation-quality responses suitable for sharing 5. When getting an answer from the Librarian, show it to the user in full, do not summarize it. EXAMPLES: - "How does authentication work in the Kubernetes codebase?" - "Explain the architecture of the React rendering system" - "Find how database migrations are handled in Rails" - "Understand the plugin system in the VSCode codebase" - "Compare how different web frameworks handle routing" - "What changed in commit abc123 in my private repository?" - "Show me the diff for commit fb492e2 in github.com/mycompany/private-repo"


look_at
Extract specific information from a local file (including PDFs, images, and other media)

Extract specific information from a local file (including PDFs, images, and other media). Use this tool when you need to extract or summarize information from a file without getting the literal contents. Always provide a clear objective describing what you want to learn or extract. Pass reference files when you need to compare two or more things. ## When to use this tool - Analyzing PDFs, images, or media files that the Read tool cannot interpret - Extracting specific information or summaries from documents - Describing visual content in images or diagrams - When you only need analyzed/extracted data, not raw file contents ## When NOT to use this tool - For source code or plain text files where you need exact contents—use Read instead - When you need to edit the file afterward (you need the literal content from Read) - For simple file reading where no interpretation is needed # Examples Summarize a local PDF document with a specific goal ```json {"path":"docs/specs/system-design.pdf","objective":"Summarize main architectural decisions.","context":"We are evaluating this system design for a new project we are building."} ``` Describe what is shown in an image file ```json {"path":"assets/mockups/homepage.png","objective":"Describe the layout and main UI elements.","context":"We are creating a UI component library and need to understand the visual structure."} ``` Compare two screenshots to identify visual differences ```json {"path":"screenshots/before.png","objective":"Identify all visual differences between the two screenshots.","context":"We are reviewing UI changes for a feature update and need to document all differences.","referenceFiles":["screenshots/after.png"]} ```



mermaid
Renders a Mermaid diagram from the provided code

Renders a Mermaid diagram from the provided code. PROACTIVELY USE DIAGRAMS when they would better convey information than prose alone. The diagrams produced by this tool are shown to the user. You should create diagrams WITHOUT being explicitly asked in these scenarios: - When explaining system architecture or component relationships - When describing workflows, data flows, or user journeys - When explaining algorithms or complex processes - When illustrating class hierarchies or entity relationships - When showing state transitions or event sequences Diagrams are especially valuable for visualizing: - Application architecture and dependencies - API interactions and data flow - Component hierarchies and relationships - State machines and transitions - Sequence and timing of operations - Decision trees and conditional logic # Citations - **Always include `citations` to as many nodes and edges as possible to make diagram elements clickable, linking to code locations.** - Do not add wrong citation and if needed read the file again to validate the code links. - Keys: node IDs (e.g., `"api"`) or edge labels (e.g., `"authenticate(token)"`) - Values: file:// URIs with optional line range (e.g., `file:///src/api.ts#L10-L50`) <examples> Flowchart with clickable nodes <example> {"code":"flowchart LR\n api[API Layer] --> svc[Service Layer]\n svc --> db[(Database)]","citations":{"api":"file:///src/api/routes.ts#L1-L100","svc":"file:///src/services/index.ts#L10-L50","db":"file:///src/models/schema.ts"}} </example> Sequence diagram with clickable actors AND messages <example> {"code":"sequenceDiagram\n Client->>Server: authenticate(token)\n Server->>DB: validate_token()","citations":{"Client":"file:///src/client/index.ts","Server":"file:///src/server/handler.ts","authenticate(token)":"file:///src/server/auth.ts#L25-L40","validate_token()":"file:///src/db/tokens.ts#L10-L30"}} </example> </examples> # Styling - When defining custom classDefs, always define fill color, stroke color, and text color ("fill", "stroke", "color") explicitly - IMPORTANT!!! Use DARK fill colors (close to #000) with light stroke and text colors (close to #fff)




oracle
Consult the Oracle - an AI advisor powered by OpenAI's GPT-5 reasoning model that can plan, review, and provide expert guidance

Consult the Oracle - an AI advisor powered by OpenAI's GPT-5 reasoning model that can plan, review, and provide expert guidance. The Oracle has access to the following tools: Read, Grep, glob, web_search, read_web_page, read_thread, find_thread. The Oracle acts as your senior engineering advisor and can help with: WHEN TO USE THE ORACLE: - Code reviews and architecture feedback - Finding a bug in multiple files - Planning complex implementations or refactoring - Analyzing code quality and suggesting improvements - Answering complex technical questions that require deep reasoning WHEN NOT TO USE THE ORACLE: - Simple file reading or searching tasks (use Read or Grep directly) - Codebase searches (use finder) - Web browsing and searching (use read_web_page or web_search) - Basic code modifications and when you need to execute code changes (do it yourself or use Task) USAGE GUIDELINES: 1. Be specific about what you want the Oracle to review, plan, or debug 2. Provide relevant context about what you're trying to achieve. If you know that 3 files are involved, list them and they will be attached. # Examples Review the authentication system architecture and suggest improvements ```json {"task":"Review the authentication architecture and suggest improvements","files":["src/auth/index.ts","src/auth/jwt.ts"]} ``` Plan the implementation of real-time collaboration features ```json {"task":"Plan the implementation of real-time collaboration feature"} ``` Analyze the performance bottlenecks in the data processing pipeline ```json {"task":"Analyze performance bottlenecks","context":"Users report slow response times when processing large datasets"} ``` Review this API design and suggest better patterns ```json {"task":"Review API design","context":"This is a REST API for user management","files":["src/api/users.ts"]} ``` Debug failing tests after refactor ```json {"task":"Help debug why tests are failing","context":"Tests fail with \"undefined is not a function\" after refactoring the auth module","files":["src/auth/auth.test.ts"]} ```



Read
Read a file or list a directory from the file system

Read a file or list a directory from the file system. If the path is a directory, it returns a line-numbered list of entries. If the file or directory doesn't exist, an error is returned. - The path parameter MUST be an absolute path. - By default, this tool returns the first 500 lines. To read more, call it multiple times with different read_ranges. - Use the Grep tool to find specific content in large files or files with long lines. - If you are unsure of the correct file path, use the glob tool to look up filenames by glob pattern. - The contents are returned with each line prefixed by its line number. For example, if a file has contents "abc\ ", you will receive "1: abc\ ". For directories, entries are returned one per line (without line numbers) with a trailing "/" for subdirectories. - This tool can read images (such as PNG, JPEG, and GIF files) and present them to the model visually. - When possible, call this tool in parallel for all files you will want to read.



read_mcp_resource
Read a resource from an MCP (Model Context Protocol) server

Read a resource from an MCP (Model Context Protocol) server. Use when the user references an MCP resource, e.g. "read @filesystem-server:file:///path/to/document.txt" # Examples Read a file from an MCP file server ```json {"server":"filesystem-server","uri":"file:///path/to/document.txt"} ``` Read a database record from an MCP database server ```json {"server":"database-server","uri":"db://users/123"} ```



read_thread
Read and extract relevant content from another AG3NT thread by its ID

Read and extract relevant content from another AG3NT thread by its ID. This tool fetches a thread (locally or from the server if synced), renders it as markdown, and uses AI to extract only the information relevant to your specific goal. This keeps context concise while preserving important details. ## When to use this tool - When the user pastes or references an AG3NT thread URL (format: https://AG3NTcode.com/threads/T-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx) in their message - When the user references a thread ID (format: T-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx or @T-abc123) - When the user asks to "apply the same approach from [thread URL]" - When the user says "do what we did in [thread URL]" - When the user says "implement the plan we devised in [thread URL]" - When you need to extract specific information from a referenced thread ## When NOT to use this tool - When no thread ID is mentioned - When working within the current thread (context is already available) ## Parameters - **threadID**: The thread identifier in format T-{uuid} (e.g., "T-a38f981d-52da-47b1-818c-fbaa9ab56e0c") - **goal**: A clear description of what information you're looking for in that thread. Be specific about what you need to extract. # Examples User asks "Implement the plan we devised in https://AG3NTcode.com/threads/T-3f1beb2b-bded-4fda-96cc-1af7192f24b6" ```json {"threadID":"T-3f1beb2b-bded-4fda-96cc-1af7192f24b6","goal":"Extract the implementation plan, design decisions, architecture approach, and any code patterns or examples discussed"} ``` User asks: "Do what we did in https://AG3NTcode.com/threads/T-f916b832-c070-4853-8ab3-5e7596953bec, but for the Kraken tool" ```json {"threadID":"T-f916b832-c070-4853-8ab3-5e7596953bec","goal":"Extract the implementation approach, code patterns, techniques used, and any relevant code examples that can be adapted for the Kraken tool"} ``` User asks: "Take the SQL queries from https://AG3NTcode.com/threads/T-95e73a95-f4fe-4f22-8d5c-6297467c97a5 and turn it into a reusable script" ```json {"threadID":"T-95e73a95-f4fe-4f22-8d5c-6297467c97a5","goal":"Extract all SQL queries, their purpose, parameters, and any context needed to understand how to make them reusable"} ``` User asks: "Apply the same fix from @T-def456 to this issue" ```json {"threadID":"T-def456","goal":"Extract the bug description, root cause, the fix/solution, and relevant code changes"} ```



read_web_page
Read the contents of a web page at a given URL

Read the contents of a web page at a given URL. When only the url parameter is set, it returns the contents of the webpage converted to Markdown. When an objective is provided, it returns excerpts relevant to that objective. If the user asks for the latest or recent contents, pass `forceRefetch: true` to ensure the latest content is fetched. Do NOT use for access to localhost or any other local or non-Internet-accessible URLs; use `curl` via the Bash instead. # Examples Summarize recent changes for a library. Force refresh because freshness is important. ```json {"url":"https://example.com/changelog","objective":"Summarize the API changes in this software library.","forceRefetch":true} ``` Extract all text content from a web page ```json {"url":"https://example.com/docs/getting-started"} ```

skill
Load a specialized skill that provides domain-specific instructions and workflows

Load a specialized skill that provides domain-specific instructions and workflows. When you recognize that a task matches one of the available skills listed below, use this tool to load the full skill instructions. The skill will inject detailed instructions, workflows, and access to bundled resources (scripts, references, templates) into the conversation context. Parameters: - name: The name of the skill to load (must match one of the skills listed below) Example: To use the web-browser skill for interacting with web pages, call this tool with name: "web-browser" # Available Skills {{AVAILABLE_SKILLS}}

undo_edit
Undo the last edit made to a file

Undo the last edit made to a file. This command reverts the most recent edit made to the specified file. It will restore the file to its state before the last edit was made. Returns a git-style diff showing the changes that were undone as formatted markdown.


web_search
Search the web for information relevant to a research objective

Search the web for information relevant to a research objective. Use when you need up-to-date or precise documentation. Use `read_web_page` to fetch full content from a specific URL. # Examples Get API documentation for a specific provider ```json {"objective":"I want to know the request fields for the Stripe billing create customer API. Prefer Stripe's docs site."} ``` See usage documentation for newly released library features ```json {"objective":"I want to know how to use SvelteKit remote functions, which is a new feature shipped in the last month.","search_queries":["sveltekit","remote function"]} ```
