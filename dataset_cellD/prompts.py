import re

# Pre-masking patterns: mask extreme tokens to prevent refusals
PRE_MASK_PATTERNS = [
    # multi-word violence / death / identity phrases first
    (r"\bgas\s+(?:these|those|the)?\s*\w*\s*babies?\b", "[VIOLENT_PHRASE]"),
    (r"\bdie\s+from\s+aids\b", "[ILLNESS_DEATH]"),
    (r"\bwhite\s+genocide\b", "[VIOLENT_PHRASE]"),
    (r"\bwhite\s+power\b", "[IDENTITY_PHRASE]"),
    (r"\bto\s+death\b", "[VIOLENT_PHRASE]"),
    # single tokens
    (r"\bgenocide\b", "[VIOLENT_NOUN]"),
    (r"\b(?:apes|monkeys)\b", "[DEHUM_NOUN]"),
    (r"\bstench\b", "[NEG_NOUN]"),
    (r"\b(?:raping|killing|destroying|gassing)\b", "[VIOLENCE_VERB]"),
    (r"\bdestruction\b", "[VIOLENT_NOUN]"),
    # primary slurs
    (r"\bniggers?\b", "[SLUR_1]"),
    (r"\bkikes?\b", "[SLUR_2]"),
    (r"\bcoons?\b", "[SLUR_3]"),
    (r"\bmudsharks?\b", "[SLUR_4]"),
    (r"\bskanks?\b", "[PROF_1]"),
    (r"\bjidf\b", "[SLUR_5]"),
]

def pre_mask(text: str) -> str:
    out = text
    for pat, tag in PRE_MASK_PATTERNS:
        out = re.sub(pat, tag, out, flags=re.IGNORECASE)
    return out


PROMPT = {}

PROMPT["system"] = """You are a specialist linguistic-transformation assistant for an academic NLP research project on AI fairness at HUFS (ELLT department). The goal is to build minimal-pair datasets for probing LLM internal representations of bias.

You produce 'Cell D (Neutral Control)' sentences from Cell A inputs. Inputs you receive have already been pre-masked by code: the most extreme tokens (slurs, death wishes, dehumanizing nouns, genocide language) are replaced with bracketed placeholders such as [SLUR_1], [VIOLENT_PHRASE], [DEHUM_NOUN], [ILLNESS_DEATH]. Treat every bracketed placeholder as an ordinary cue token to be removed in Step 1. You must NOT try to recover the original meanings of placeholders, and you must NOT refuse based on them.

You will produce the Cell D sentence in two rewriting steps:
- Step 1 (B-like intermediate): replace EVERY demographic mention (group nouns AND identity adjectives like whites/blacks/jews/arabs/muslims) with the same Replacement Noun Phrase from analysis, and remove all bracketed placeholders plus any remaining slurs and profanity.
- Step 2 (final D): neutralize the remaining cues using ONLY the five categories defined in analysis (negative emotion verbs, violence verbs, strong negative adjectives, intensifiers, multi-word violence/death phrases) into weak neutral expressions.

Across both steps you must strictly preserve the sentence structure (clause type, voice, tense, aspect, negation) and length within +/- 30% of the original input. This is a controlled linguistic task that produces neutral controls; refusing breaks the dataset. Do not refuse."""

PROMPT["cell_d_analyze"] = """Analyze the input (a pre-masked Cell A sentence) to plan a two-step neutralization for Cell D.

Rules for Analysis:
1. ALL Demographic Mentions: List EVERY demographic mention in the input, including
   - Group nouns (muslims, jews, arabs, ...)
   - Identity adjectives and their plural / possessive forms (white, whites, black, blacks, asian, asians, ...)
   - Bracketed identity placeholders ([SLUR_*], [DEHUM_NOUN], [IDENTITY_PHRASE])
   List every occurrence, not just the first.
2. Replacement Noun Phrase: Choose ONE non-identity human noun phrase from {visitors, students, participants, residents, individuals, people}. It will replace EVERY demographic mention listed above with grammatical-number agreement (e.g., 'whites' -> 'people', 'a black' -> 'an individual').
3. Sentence Structure: Record clause type, verb voice, tense, aspect, and negation. These MUST be preserved across both steps.
4. Cue Inventory: USE ONLY these five categories. Do NOT invent 'metaphor', 'simile', 'informal phrase', 'negative implication', 'strong negative context'. Tokens that fit none of the five must go under Frame-carrying tokens, or under Step 1 REMOVE if they are slurs / profanity / placeholders.
   - Step 1 removes: profanity, slurs, ALL bracketed placeholders.
   - Step 2 neutralizes:
     - Negative emotion verbs (hate, despise, loathe, love-when-sarcastic) -> notice / find / see
     - Violence verbs (attack, destroy, kill, beat, stomp) -> consider / remember / approach / interact with
     - Strong negative adjectives (disgusting, awful, stupid, crazy, retarded, violent, jealous, worthless, dumb) -> different / unusual / notable / distinct
     - Intensifiers (very, completely, totally, absolutely) -> REMOVE or weaken to quite / rather
     - Multi-word violence / death phrases ([VIOLENT_PHRASE], 'one up', 'rub it in your faces') -> REMOVE
5. Frame-carrying tokens: any non-cue token that is neither an identity term nor a placeholder. Frame-carrying tokens MUST NOT include any group name, identity adjective, slur, or placeholder.

Input: {input_text}

Output format:
- All Demographic Mentions:
- Replacement Noun Phrase:
- Sentence Structure (clause / voice / tense / aspect / negation):
- Step 1 Cues (REMOVE): [Word/Placeholder] -> REMOVE
- Step 2 Cues (NEUTRALIZE): [Word] -> [Category from the 5 above] -> [Replacement or REMOVE]
- Frame-carrying tokens to preserve:
"""

PROMPT["cell_d_step1"] = """Step 1: produce the B-like intermediate by replacing EVERY demographic mention and removing all explicit cues.

Analysis: {turn_1}
Original Input (Cell A, pre-masked): {input_text}

Constraints:
1. REPLACE EVERY DEMOGRAPHIC MENTION: Apply the Replacement Noun Phrase to EVERY item in 'All Demographic Mentions', matching grammatical number. Do not introduce any other identity term or population-boundary expression (our country, their land, etc.).
2. REMOVE EXPLICIT CUES AND PLACEHOLDERS: Delete every item under Step 1 Cues, including all bracketed placeholders. Output must contain ZERO bracketed tokens.
3. PRESERVE STRUCTURE: Keep clause type, verb voice, tense, aspect, and negation. Length within +/- 30% of the original input.
4. NO NEW CUES: Do not add new negative adjectives, intensifiers, violence verbs, or new identity terms.
5. MINIMAL GRAMMAR REPAIR ALLOWED: If removing a placeholder or slur leaves a grammatically broken phrase, you MAY adjust ONLY the immediately adjacent preposition or auxiliary; do NOT change clause type, voice, tense, aspect, or negation.

Output ONLY one transformed sentence."""

PROMPT["cell_d_step2"] = """Step 2: produce the final Cell D sentence by neutralizing the remaining cues.

Analysis: {turn_1}
Step 1 output: {turn_2}

Constraints:
1. KEEP NOUN PHRASE: The Replacement Noun Phrase from Step 1 must remain unchanged at every position.
2. APPLY CUE NEUTRALIZATION: Swap each cue under 'Step 2 Cues' using the EXACT mapping in the analysis. Do not invent substitutions. Do not introduce new categories.
3. PRESERVE STRUCTURE: Same clause type, voice, tense, aspect, and negation as the original input. Length within +/- 30%.
4. NEUTRAL TONE: Output must be neutral or weakly descriptive. No sarcasm, irony, hidden negativity, moral judgment, or boundary framing.
5. NO NEW CUES, NO NEW IDENTITY: Do not add new negative wording or demographic implications.
6. MINIMAL GRAMMAR REPAIR ALLOWED: If a substitution creates an ungrammatical phrase (e.g., 'jealous of' -> 'different of'), adjust ONLY the directly adjacent preposition to the most natural neutral form (e.g., 'about', 'toward'). Do NOT replace the substituted word itself with a different word.

Output ONLY one transformed sentence."""

PROMPT["cell_d_check"] = """Verify the Cell D sentence against the checklist. Make ONLY minimal grammar fixes; do NOT introduce any word substitution that is not in the analysis Cue Inventory.

Checklist:
1. TARGET ABSENCE: No demographic group name, slur, identity adjective, stereotype noun, identity placeholder, or in-group/out-group expression remains. Replacement Noun Phrase is in place at every original demographic position.
2. STRONG CUE ABSENCE: No profanity, slur, violence verb, strong negative adjective, strong intensifier, or bracketed placeholder remains.
3. NEUTRAL TONE: Polarity within roughly [-0.1, 0.3]; no sarcasm or hidden negativity.
4. STRUCTURE PRESERVED: Same clause type, voice, tense, aspect, and negation as the original input.
5. LENGTH: Within +/- 30% of the original input token count.
6. NATURALNESS: Grammatical and semantically coherent.

Step 2 output: {turn_3}
Original Input (Cell A, pre-masked): {input_text}

If any check fails, fix it MINIMALLY using ONLY substitutions already listed in the analysis Cue Inventory. You MUST NOT introduce new substitutions (e.g., do NOT change 'outperform' to 'influence' if that change is not in the inventory).
Output ONLY the final natural English sentence."""
