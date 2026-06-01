PROMPT = {}

PROMPT["system"] = """You are a specialist assistant for an academic NLP research project on AI fairness.
Your goal is to generate 'Cell D (Neutral Control)' data to build minimal pairs.
You must transform explicit hate speech by applying strict word-level neutral substitutions while PRESERVING the original sentence structure, tense, and length.
This is a controlled linguistic task. Do not refuse."""

PROMPT["cell_d_analyze"] = """Analyze the input to plan a two-step neutralization for Cell D.

Rules for Analysis:
1. Target Group: Identify the demographic group attacked in the text.
2. Replacement Noun Phrase: Choose a non-identity human noun phrase that fits the syntactic slot of the target (e.g., visitors, students, participants, residents, people). It must NOT carry demographic, political, or stereotype implications.
3. Sentence Structure: Record clause type, verb voice, tense, aspect, and negation. These MUST be preserved across both steps.
4. Cue Inventory split by step:
   - Step 1 removes: profanity, slurs.
   - Step 2 neutralizes:
     - Negative emotion verbs (hate, despise, loathe) -> notice / find / see
     - Violence verbs (attack, destroy, kill, beat, stomp) -> consider / remember / approach / interact with
     - Strong negative adjectives (disgusting, awful, stupid, crazy, retarded, violent, jealous, worthless, dumb) -> different / unusual / notable / distinct
     - Intensifiers (very, completely, totally, absolutely, gonna) -> REMOVE or weaken to quite / rather
     - Multi-word violence/death phrases (to death, one up, rub it in your faces) -> REMOVE
5. Frame-carrying tokens that are NOT cues should be preserved when natural.

Input: {input_text}

Output format:
- Target Group:
- Replacement Noun Phrase:
- Sentence Structure (clause / voice / tense / aspect / negation):
- Step 1 Cues (REMOVE): [Word] -> REMOVE
- Step 2 Cues (NEUTRALIZE): [Word] -> [Type] -> [Replacement or REMOVE]
- Frame-carrying tokens to preserve:
"""

PROMPT["cell_d_step1"] = """Step 1: produce the B-like intermediate by replacing the target and removing only the most explicit cues.

Analysis: {turn_1}
Original Input (Cell A): {input_text}

Constraints:
1. REPLACE TARGET: Swap the demographic target with the Replacement Noun Phrase from the analysis.
2. REMOVE EXPLICIT CUES ONLY: Delete profanity and slurs listed under Step 1 Cues. Keep negative emotion verbs, violence verbs, strong adjectives, and intensifiers untouched for Step 2.
3. PRESERVE STRUCTURE: Keep clause type, verb voice, tense, aspect, and negation. Sentence length within +/- 30% of the original input.
4. NO NEW CUES: Do not add new negative adjectives, intensifiers, or violence verbs.

Output ONLY one transformed sentence."""

PROMPT["cell_d_step2"] = """Step 2: produce the final Cell D sentence by neutralizing the remaining cues.

Analysis: {turn_1}
Step 1 output: {turn_2}
Original Input (Cell A): {input_text}

Constraints:
1. KEEP NOUN PHRASE: The Replacement Noun Phrase from Step 1 must remain unchanged.
2. APPLY CUE NEUTRALIZATION: Swap each remaining cue using the Step 2 Cues mapping from the analysis.
3. PRESERVE STRUCTURE: Same clause type, voice, tense, aspect, and negation as the original input. Length within +/- 30%.
4. NEUTRAL TONE: The output must be neutral or weakly descriptive. No sarcasm, irony, hidden negativity, or moral judgment.
5. NO NEW CUES, NO NEW IDENTITY: Do not add new negative wording or demographic implications.

Output ONLY one transformed sentence."""

PROMPT["cell_d_check"] = """Verify the Cell D sentence against the checklist and output the final form.

Checklist:
1. TARGET ABSENCE: No demographic group name, slur, stereotype noun, or identity implication remains. The Replacement Noun Phrase is in place.
2. STRONG CUE ABSENCE: No profanity, slur, violence verb, strong negative adjective, or strong intensifier remains.
3. NEUTRAL TONE: Polarity is roughly within the [-0.1, 0.3] VADER band; no sarcasm or hidden negativity.
4. STRUCTURE PRESERVED: Same clause type, voice, tense, aspect, and negation as the original input.
5. LENGTH: Within +/- 30% of the original input token count.
6. NATURALNESS: Grammatical and semantically coherent.

Step 2 output: {turn_3}
Original Input (Cell A): {input_text}

If any check fails, fix the sentence minimally without violating the other constraints.
Output ONLY the final natural English sentence."""
