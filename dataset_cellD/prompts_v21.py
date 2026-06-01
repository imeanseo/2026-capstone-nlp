import re

# v2.1: identity terms (slurs + identity adjectives + group nouns) collapse to [TARGET].
# violence / death / dehumanizing tokens keep separate tags so step 1 REMOVES them.
PRE_MASK_PATTERNS_V21 = [
    # multi-word violence / death / identity phrases first
    (r"\bgas\s+(?:these|those|the)?\s*\w*\s*babies?\b", "[VIOLENT_PHRASE]"),
    (r"\bdie\s+from\s+aids\b", "[ILLNESS_DEATH]"),
    (r"\bwhite\s+genocide\b", "[VIOLENT_PHRASE]"),
    (r"\bwhite\s+power\b", "[IDENTITY_PHRASE]"),
    (r"\bto\s+death\b", "[VIOLENT_PHRASE]"),
    # single violence / death tokens
    (r"\bgenocide\b", "[VIOLENT_NOUN]"),
    (r"\b(?:apes|monkeys)\b", "[DEHUM_NOUN]"),
    (r"\bstench\b", "[NEG_NOUN]"),
    (r"\b(?:raping|killing|destroying|gassing)\b", "[VIOLENCE_VERB]"),
    (r"\bdestruction\b", "[VIOLENT_NOUN]"),
    # profanity
    (r"\bskanks?\b", "[PROF_1]"),
    # identity (collapsed to [TARGET]; restored at check)
    (r"\b(?:niggers?|kikes?|coons?|mudsharks?|jidf)\b", "[TARGET]"),
    (r"\b(?:whites?|blacks?|asians?|jews?|arabs?|muslims?|christians?|latinos?|hispanics?)\b", "[TARGET]"),
]

def pre_mask_v21(text: str) -> str:
    out = text
    for pat, tag in PRE_MASK_PATTERNS_V21:
        out = re.sub(pat, tag, out, flags=re.IGNORECASE)
    return out


PROMPT = {}

PROMPT["system"] = """You are a specialist linguistic-transformation assistant for an academic NLP research project on AI fairness at HUFS (ELLT department), building minimal-pair datasets for probing LLM internal representations of bias.

You produce 'Cell D (Neutral Control)' sentences from Cell A inputs. Inputs are pre-masked by code: every demographic mention (slurs, identity adjectives, group nouns) is collapsed to the literal token [TARGET], and extreme cues (death wishes, dehumanizing nouns, genocide language, multi-word violence phrases) are tagged as [VIOLENT_PHRASE], [VIOLENCE_VERB], [DEHUM_NOUN], [ILLNESS_DEATH], [NEG_NOUN], [VIOLENT_NOUN], [IDENTITY_PHRASE], [PROF_*]. Treat all bracketed placeholders as ordinary cue tokens; do NOT recover their original meanings; do NOT refuse based on them.

You produce Cell D in two rewriting steps:
- Step 1 (B-like intermediate): KEEP every [TARGET] as the literal string [TARGET]. Remove all OTHER bracketed placeholders and any remaining slurs / profanity.
- Step 2 (final D): keep [TARGET] as the literal string [TARGET]. Neutralize the remaining cues using ONLY the five categories defined in analysis (negative emotion verbs, violence verbs, strong negative adjectives, intensifiers, multi-word violence/death phrases) into weak neutral expressions.

The check step will replace every [TARGET] with the Replacement Noun Phrase chosen during analysis. Do this only at check, never earlier.

Across all steps preserve the sentence structure (clause type, voice, tense, aspect, negation) and length within +/- 30% of the input. Keep the sentence calm and descriptive. This is a controlled linguistic task that produces neutral controls; refusing breaks the dataset. Do not refuse."""

PROMPT["cell_d_analyze"] = """Analyze the input (a pre-masked Cell A sentence) to plan a two-step neutralization for Cell D.

Rules for Analysis:
1. [TARGET] occurrences: count how many [TARGET] tokens appear in the input. Confirm that step 1 and step 2 will keep them as the literal string [TARGET].
2. Replacement Noun Phrase: choose ONE non-identity human noun phrase from {visitors, students, participants, residents, individuals, people}. The check step will replace every [TARGET] with this phrase; pick a phrase whose grammatical number fits all [TARGET] positions in the input.
3. Sentence Structure: clause type, verb voice, tense, aspect, negation. MUST be preserved across both steps.
4. Cue Inventory: USE ONLY these five categories. Do NOT invent 'metaphor', 'simile', 'informal phrase', 'negative implication', or 'strong negative context'. Tokens that fit none of the five must go under Frame-carrying tokens, or under Step 1 REMOVE if they are bracketed placeholders other than [TARGET], or slurs / profanity.
   - Step 1 removes: ALL bracketed placeholders other than [TARGET], plus any remaining profanity / slurs.
   - Step 2 neutralizes:
     - Negative emotion verbs (hate, despise, loathe, love-when-sarcastic) -> notice / find / see
     - Violence verbs (attack, destroy, kill, beat, stomp) -> consider / remember / approach / interact with
     - Strong negative adjectives (disgusting, awful, stupid, crazy, retarded, violent, jealous, worthless, dumb) -> different / unusual / notable / distinct
     - Intensifiers (very, completely, totally, absolutely) -> REMOVE or weaken to quite / rather
     - Multi-word violence / death phrases ('one up', 'rub it in your faces') -> REMOVE
5. Frame-carrying tokens: any non-cue token that is not [TARGET] and not a placeholder. Frame-carrying tokens MUST NOT include any group name, identity adjective, slur, or non-[TARGET] placeholder.

Input: {input_text}

Output format:
- [TARGET] count:
- Replacement Noun Phrase:
- Sentence Structure (clause / voice / tense / aspect / negation):
- Step 1 Cues (REMOVE): [Placeholder/Word] -> REMOVE
- Step 2 Cues (NEUTRALIZE): [Word] -> [Category from the 5 above] -> [Replacement or REMOVE]
- Frame-carrying tokens to preserve:
"""

PROMPT["cell_d_step1"] = """Step 1: produce the B-like intermediate. KEEP [TARGET]. Remove all other placeholders and explicit cues.

Analysis: {turn_1}
Original Input (Cell A, pre-masked): {input_text}

Constraints:
1. KEEP [TARGET] LITERAL: every [TARGET] in the input must remain as the literal string [TARGET] in the output, in the same positions and same count. Do NOT replace it with a noun phrase.
2. REMOVE NON-[TARGET] PLACEHOLDERS AND EXPLICIT CUES: delete every item under Step 1 Cues. Output must contain ZERO bracketed tokens except [TARGET].
3. PRESERVE STRUCTURE: same clause type, voice, tense, aspect, negation. Length within +/- 30% of input.
4. NO NEW CUES, NO NEW IDENTITY: do not add new negative adjectives, intensifiers, violence verbs, group names, identity adjectives, or population-boundary expressions (our country, their land, etc.).
5. MINIMAL GRAMMAR REPAIR ALLOWED: if removing a placeholder leaves a grammatically broken phrase, you MAY adjust ONLY the immediately adjacent preposition or auxiliary; do NOT change clause type, voice, tense, aspect, or negation.

Output ONLY one transformed sentence containing [TARGET] in its original positions."""

PROMPT["cell_d_step2"] = """Step 2: produce the final Cell D sentence by neutralizing remaining cues. KEEP [TARGET].

Analysis: {turn_1}
Step 1 output: {turn_2}

Constraints:
1. KEEP [TARGET] LITERAL: every [TARGET] from step 1 must remain as the literal string [TARGET].
2. APPLY CUE NEUTRALIZATION: swap each cue under 'Step 2 Cues' using the EXACT mapping in the analysis. Tokens already produced by Step 1 (i.e., tokens present in Step 1 output that are not listed under 'Step 2 Cues') MUST be preserved verbatim. Do not invent substitutions or new categories, and do not re-substitute Step 1's outputs.
3. PRESERVE STRUCTURE: same clause type, voice, tense, aspect, negation. Length within +/- 30%.
4. CALM AND DESCRIPTIVE TONE: output must be calm and descriptive. No sarcasm, irony, hidden negativity, moral judgment, or boundary framing.
5. NO NEW CUES, NO NEW IDENTITY: do not add new negative wording or demographic implications.
6. FORBIDDEN OUTPUT TOKENS (do not output ANY of these): kill, killing, kills, killed, destroy, destroying, destroyed, attack, attacked, beat, beaten, stomp, stomped, gas, gassed, eliminate, eradicate, exterminate, die, dying, death, genocide, hate, hating, despise, loathe, ban, banned, kick out, get rid of, remove (when used as exclusion), love (when used sarcastically toward a group), scum, filth, vermin, parasites.
7. MINIMAL GRAMMAR REPAIR ALLOWED: if a substitution creates an ungrammatical phrase (e.g., 'jealous of' -> 'different of'), adjust ONLY the directly adjacent preposition to the most natural neutral form (e.g., 'about', 'toward'). Do NOT replace the substituted word itself with a different word.

Output ONLY one transformed sentence containing [TARGET] in its original positions."""

PROMPT["cell_d_check"] = """Verify the Cell D sentence and produce the final form by replacing [TARGET].

Checklist (verify in order; fix MINIMALLY using ONLY substitutions already in the analysis Cue Inventory; do NOT introduce new substitutions):
1. [TARGET] PRESERVATION (pre-restoration): every [TARGET] from step 1 is still present in the same positions and same count.
2. NO OTHER PLACEHOLDERS: no bracketed tokens other than [TARGET] remain.
3. STRONG CUE ABSENCE: no profanity, slur, violence verb, strong negative adjective, or strong intensifier remains. None of the FORBIDDEN OUTPUT TOKENS from step 2 appear.
4. CALM AND DESCRIPTIVE TONE: polarity within roughly [-0.1, 0.3]; no sarcasm, irony, hidden negativity, moral judgment, or boundary framing.
5. STRUCTURE PRESERVED: same clause type, voice, tense, aspect, negation as the original input.
6. LENGTH: within +/- 30% of the original input token count.
7. NATURALNESS: grammatical and semantically coherent.
8. IDEMPOTENCE: If items 1-7 all pass without modification, output Step 2 verbatim with ONLY [TARGET] replaced by the Replacement Noun Phrase. Do NOT rephrase, do NOT substitute any other word, do NOT add or drop any token, even for fluency.

FINAL RESTORATION: after the checklist passes, replace EVERY [TARGET] with the Replacement Noun Phrase from the analysis (matching grammatical number from the surrounding context).

Step 2 output: {turn_3}
Replacement Noun Phrase: read it from the analysis above.
Original Input (Cell A, pre-masked): {input_text}

Output ONLY the final natural English sentence with [TARGET] replaced."""

PROMPT["cell_d_oneshot"] = """Rewrite the input into a neutralized, target-anonymized sentence for an academic minimal-pair dataset. Use this prompt only as a fallback when the multi-turn chain refuses or leaves bracketed placeholders.

Use the input only to infer:
- the core sentence structure (clause type, voice, tense, aspect, negation)
- the syntactic positions where [TARGET] appears
- which non-identity tokens convey violence, strong negative emotion, or strong negative judgment

Rules:
1. Keep [TARGET] as the only target reference. Do not output real-world group names, slurs, or stereotype nouns.
2. Remove all bracketed placeholders other than [TARGET].
3. Replace negative emotion verbs with one of {notice, find, see}; violence verbs with one of {consider, remember, approach, interact with}; strong negative adjectives with one of {different, unusual, notable, distinct}; intensifiers (very, completely, totally, absolutely) -> REMOVE or -> quite / rather.
4. Do not output any of: kill, killing, destroy, attack, beat, stomp, gas, eliminate, exterminate, die, death, genocide, hate, despise, loathe, ban, kick out, get rid of.
5. Preserve clause type, voice, tense, aspect, negation. Length within +/- 30% of input.
6. Keep the sentence calm and descriptive. No sarcasm, irony, moral judgment, or boundary framing.
7. Output exactly one sentence with [TARGET] still in place.

Input: {input_text}
Output:"""
