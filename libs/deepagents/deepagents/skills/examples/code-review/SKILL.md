---
id: code-review
name: Code Review
description: Performs thorough code review with focus on correctness, security, performance, and maintainability. Provides actionable feedback with specific line references.
version: "1.0.0"
mode: both
tags: ["code", "review", "quality", "security"]
tools: ["view", "codebase-retrieval", "diagnostics"]
inputs: File paths or code snippets to review
outputs: Structured review with findings categorized by severity
safety: Read-only analysis, no code modifications
triggers:
  - "review this code"
  - "check for bugs"
  - "security review"
  - "code quality check"
---

## Purpose

This skill performs comprehensive code review, analyzing code for:
- Correctness and potential bugs
- Security vulnerabilities
- Performance issues
- Code style and maintainability
- Best practices adherence

## When to Use

Use this skill when:
- Reviewing pull requests or code changes
- Auditing code for security vulnerabilities
- Checking code quality before deployment
- Learning from code patterns and anti-patterns

## Operating Procedure

1. **Gather Context**: Use `codebase-retrieval` to understand the codebase structure and related files.

2. **Load Files**: Use `view` to read the files to be reviewed.

3. **Check Diagnostics**: Use `diagnostics` to get IDE-reported issues.

4. **Analyze Code**: Review for:
   - Logic errors and edge cases
   - Null/undefined handling
   - Resource leaks
   - Race conditions
   - Input validation
   - Authentication/authorization issues
   - SQL injection, XSS, and other security issues
   - Performance bottlenecks
   - Code duplication
   - Naming and documentation

5. **Categorize Findings**: Group by severity:
   - 🔴 Critical: Security vulnerabilities, data loss risks
   - 🟠 High: Bugs that will cause failures
   - 🟡 Medium: Code smells, maintainability issues
   - 🟢 Low: Style suggestions, minor improvements

6. **Provide Recommendations**: For each finding, provide:
   - Location (file:line)
   - Description of the issue
   - Why it matters
   - Suggested fix

## Tool Usage Rules

- Use `view` to read source files
- Use `codebase-retrieval` to find related code and understand context
- Use `diagnostics` to get type errors and linting issues
- Do NOT use any tools that modify files
- Do NOT use shell commands

## Output Format

```markdown
# Code Review: [File/Component Name]

## Summary
Brief overview of the review findings.

## Critical Issues 🔴
### Issue 1: [Title]
- **Location**: `file.py:42`
- **Description**: ...
- **Impact**: ...
- **Recommendation**: ...

## High Priority 🟠
...

## Medium Priority 🟡
...

## Low Priority 🟢
...

## Positive Observations ✅
- Good patterns observed
- Well-implemented features
```

## Failure Modes and Recovery

1. **File not found**: Request correct file path from user.
2. **Large codebase**: Focus on changed files or specific modules.
3. **Unfamiliar language**: Note limitations and focus on general patterns.
4. **Missing context**: Use codebase-retrieval to gather more context.

