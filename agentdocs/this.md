# This

How every file under `agentdocs/` is written. This file is itself an example of the format.

## File shape

**One topic per file. Start with a title and a one-sentence description.**

- `# Title`: one word or short phrase, no trailing punctuation.
- One sentence under the title that states what the file covers.
- Sections use plain headings: `## Title Cased Heading`. No numbers: adding a section must not force renumbering.
- Each section is self-contained; a reader can skip to it directly.

## Section shape

**Bold rule first. Bullets second. A closing sentence is optional.**

- The first line after the heading is a bold one-liner stating the rule.
- The rule is an instruction, not a description.
- Bullets follow, optionally preceded by a one-line framing sentence.
- A closing sentence is added only when it carries information the bullets do not.

## Bullets

**Three to five bullets per section. One line each. Imperative voice.**

- Start with a capital letter; end with a period.
- Lead with the verb or with the thing being forbidden.
- Two short sentences per bullet are acceptable; longer bullets are not.
- Nested bullets are used only under a parent line ending in a colon.

## Enumerations

**Use bullets, not tables.**

- Tables produce wide rows that are hard to compare.
- For `name: description` pairs, write `` `Name`: description. ``
- Group related bullets under a one-line header ending in a colon.
- Code fences are acceptable for commands and small code examples.

## Punctuation

**Colons, not em dashes.**

- Use `:` where an em dash would otherwise appear.
- Use commas or parentheses for short parenthetical asides.
- `>` blockquotes are reserved for callouts at the top of a file.

## Voice

**Direct and neutral. No marketing language. No unnecessary jargon.**

- State the rule; justify only when the rule is not obvious on its own.
- Prefer present tense and second person over passive voice.
- Avoid adjectives that do not carry information ("powerful", "clean", "seamless").
- Avoid borrowed metaphors ("kernel", "plane", "seam", "pipeline") unless they are the precise technical term.

## Emphasis

**Use MUST for non-negotiable rules. Use IMPORTANT for easy-to-miss gotchas.**

- MUST: correctness-critical rules where a violation breaks compilation, the wire protocol, or an architectural invariant.
- IMPORTANT: prefixes a bullet that a reader skimming would miss and regret later.
- Most rules need neither: the bold one-liner is already the rule.
- SHOULD, MAY, and CAN are not used: RFC-2119 without the full spec is noise.

## Code grounding

**Rules name identifiers that exist in the crate. No invented vocabulary.**

- A type, function, field, or method named in a rule MUST be greppable in `crates/agentwerk/src/`.
- Verbs describe what the code does, not how it feels: avoid "wires", "magic", "ergonomic", "seamless".
- A rule that cannot point at code is opinion, not architecture: drop it or move it to the consuming application.
- When a name changes in code, the docs change in the same commit.

## Cross-linking

**Each fact lives in one file. Other files link to it.**

- Commands belong in `workflow.md`; other files link to it rather than restating commands.
- File and module placement belongs in `layout.md`; `architecture.md` describes invariants and assumes placement is known.
- Naming and comment rules belong in `style.md`; `testing.md` covers test-specific naming and links out for the rest.
- A duplicated fact is a future inconsistency: when two files would say the same thing, one of them links instead.

## Length

**If agentdocs are getting too long, consolidate them. Information loss is acceptable.**

- Drop sections that restate what a careful reader of the code would already see.
- Merge two short, overlapping sections before splitting one long section.
- A rule that has not earned its line is cut, not rewritten shorter.
- Skim cost matters more than completeness: a forgotten file teaches nothing.
