---
description: Generate a validated cheatsheet for a Python package
argument-hint: <package_name> [description of use cases to cover]
allowed-tools: Bash, Read, Write, Edit
---

Parse `$ARGUMENTS` as follows: the first word is the Python package name to
document; everything after it is an optional description of the use cases to
cover. Assign these to `PACKAGE` and `DESCRIPTION` respectively.

If no description was given, run the package's own help to infer what it does:

```bash
/usr/bin/python3 -c "import $1; help($1)" 2>/dev/null | head -40
```

Then follow all steps below.

---

## Step 1 — Inspect the package before writing anything

Do NOT rely on memory for any function name, argument, or import path.
Run these first and read the output:

```bash
# Top-level API
/usr/bin/python3 -c "import $1; print([x for x in dir($1) if not x.startswith('_')])"

# If the package has submodules, inspect them too
/usr/bin/python3 -c "
import $1, inspect
for name, obj in inspect.getmembers($1):
    if inspect.ismodule(obj):
        print(f'Submodule: {name}')
        print([x for x in dir(obj) if not x.startswith('_')])
"

# Check the package version so it's noted in the cheatsheet
/usr/bin/python3 -c "import $1; print(getattr($1, '__version__', 'version unknown'))"
```

For any function you plan to include, verify its signature:

```bash
/usr/bin/python3 -c "import inspect, $1; help($1.FUNCTION_NAME)"
/usr/bin/python3 -c "import inspect, $1; print(inspect.signature($1.FUNCTION_NAME))"
```

If a name doesn't exist, find the right one:

```bash
/usr/bin/python3 -c "import $1; print([x for x in dir($1) if 'KEYWORD' in x.lower()])"
```

**Rule:** if `hasattr(module, 'name')` is False, do not include it.

---

## Step 2 — Write the cheatsheet

Save to `docs/$1_reference.md`. Follow this structure:

```
# <Package Name> Quick Reference
<!-- version X.Y.Z -->

## 1. Installation
(bash block)

## 2. Authentication / Setup
(only if the package requires credentials — verify the exact function name
before writing it; explain where to get credentials)

## 3. <First main use case from DESCRIPTION or inferred from docs>
(1–2 sentence explanation, then a ```python block)

## 4. <Second use case>
...

## N. Complete Minimal Workflow
(Single ```python block that imports → sets up → runs a realistic end-to-end
example using only functions verified in Step 1)
```

**Format rules** (enforced by the validator):

- All Python goes in fenced ` ```python ` blocks, bash in ` ```bash ` blocks
- Put ALL top-level imports in the very first `python` block
- Later blocks may use names imported in earlier blocks (like a notebook)
- Annotate non-obvious calls with `# comments`
- If a call needs credentials or network access, add `# requires auth` but
  still include it — attribute existence is checked without credentials

---

## Step 3 — Validate and fix

```bash
/usr/bin/python3 validate_cheatsheet.py docs/$1_reference.md
```

If the validator reports any errors, go back to Step 1 and find the correct
name using `dir()` or `help()` — do not guess. Repeat until exit code is 0.

If `validate_cheatsheet.py` is not present in this project, say so and
show the user where to get it rather than skipping validation.
