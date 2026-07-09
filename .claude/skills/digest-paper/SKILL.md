---
name: digest-paper
description: Summarize a scientific PDF into structured notes, saved to references/summaries/ with an entry added to references/INDEX.md. Run once per paper.
invocation: user
argument-hint: <path/to/paper.pdf>
allowed-tools: Bash, Read, Write, Edit
---

The PDF to process is: `$ARGUMENTS`

Follow each step in order. Do not skip steps or reorder them.

---

## Step 1 — Extract the paper text

```bash
pdftotext "$ARGUMENTS" -
```

Read the full output carefully. If the result is mostly garbled or empty (common
with scanned PDFs), stop and tell the user — pdftotext cannot handle scanned
documents without OCR and the summary would be unreliable.

---

## Step 2 — Read project context

```bash
cat CLAUDE.md
```

Note the project's goals, methods, key concepts, and open questions. You will
use this in Step 4 to write the "Relevance to This Project" section. If
CLAUDE.md does not exist, note that and leave the relevance section blank.

---

## Step 3 — Determine output paths

From the PDF filename, derive a slug:
- Strip the directory path and `.pdf` extension
- Lowercase the result
- Replace spaces and hyphens with underscores

Examples:
- `references/pdfs/Card1996_MothSearch.pdf` → `card1996_mothsearch`
- `references/pdfs/Wystrach 2020.pdf`       → `wystrach_2020`

Output paths:
- Summary : `references/summaries/{slug}.md`
- Index   : `references/INDEX.md`

Create the summaries directory if needed:
```bash
mkdir -p references/summaries
```

If a summary file for this slug already exists, tell the user and ask whether
to overwrite it before continuing.

---

## Step 4 — Write the structured summary

Save to `references/summaries/{slug}.md` using exactly this template.
Fill every section from the extracted text. If a field genuinely cannot be
determined from the paper (e.g. no DOI listed), write `N/A`.

```markdown
# {Full Title}

**Authors:** {Last, First Initial; Last, First Initial; ...}
**Year:** {YYYY}
**Journal / Venue:** {Journal name, volume, pages}
**DOI / URL:** {doi or URL if present in the paper}

---

## Key Finding
{The single most important result or claim, in 1–2 sentences. Write this as a
standalone statement someone could read without context.}

## Background & Motivation
{What problem or question does this paper address? Why does it matter?
2–4 sentences.}

## Methods
{How was the study conducted? Include: study design (experimental/observational/
theoretical/computational), model system or species if applicable, key
techniques or instruments, sample sizes or dataset scale, and any important
conditions or controls. Use bullet points if there are multiple distinct
methodological components.}

## Main Results
{Bullet points of the primary findings, in order of importance. Include
quantitative results where reported (effect sizes, p-values, etc.).}

## Evidence Strength
{Characterize the quality of the evidence: Is this causal or correlational?
Is the finding replicated or novel? Are sample sizes adequate? Is the study
computational, experimental, or a review/meta-analysis?}

## Limitations & Caveats
{What are the most important constraints on interpreting or generalizing these
results? Include both limitations the authors acknowledge and any you identify.}

## Open Questions
{What does this paper leave unresolved? What follow-up work does it suggest?}

## Relevance to This Project
{Written using context from CLAUDE.md: how do the findings, methods, or
framing of this paper connect to the current project's goals or open questions?
Be specific. If no clear connection exists, say so rather than inventing one.}

## Keywords
{6–10 keywords, comma-separated, lowercase}
```

---

## Step 5 — Update the index

Open `references/INDEX.md` (create it with a header if it does not exist).
Append the following entry at the bottom. Do not modify existing entries.

```markdown
### {LastName}{Year} — {Short title, max 8 words}
{One paragraph, 3–5 sentences: the key finding, the methods in brief, and
why it is in this reference library. Write it so someone skimming the index
can decide whether to read the full summary.}
→ `references/summaries/{slug}.md`

```

If `references/INDEX.md` does not yet exist, create it with this header first:

```markdown
# Reference Index

Summaries of papers relevant to this project.
Each entry links to a full structured summary in `references/summaries/`.

---

```

---

## Step 6 — Confirm completion

Report back:
- The slug used
- The path the summary was saved to
- Whether INDEX.md was created fresh or appended to
- Any fields that could not be filled from the paper text
