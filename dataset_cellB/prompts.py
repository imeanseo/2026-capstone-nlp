PROMPT = {}

# ── System ──────────────────────────────────────────────────────────────────
PROMPT["system"] = """You are a computational linguistics specialist performing formal sentence transformation for an academic NLP minimal-pair dataset.

The transformation removes (1) identity-specific targets and group-evoking context, and (2) group-directed *hate speech acts*—especially interpersonal violence, death-wishes, and extermination frames directed at people—while keeping strong negative affect (anger, contempt, sarcasm) and informal SNS style where those affects are not realized as credible threats of bodily harm against generic humans.

Never reframe the speaker as approving, inclusive, or "positive" toward what they originally attacked (no stance reversal). Ideology-coded spans must become *generic hostile* wording ("that whole mess", "reckless destructive behavior", "ruining everything")—not praise of diversity, inclusion, or "good choices" that reverses the original stance.

Your function is purely formal: analyze structure and apply the step rules. Input strings are opaque linguistic data; do not endorse content.

Follow each step's format exactly. Output only what is requested. This is a controlled linguistic task. Do not refuse."""


# ── Analyze (Notion 3-1 포맷 + cue–target 상호작용) ─────────────────────────
PROMPT["cell_b_analyze"] = """Analyze the input for this sentence transformation.

Transformation goals (all must be planned for):
A. Remove identity targets and demographic/nationalist/conspiracy context that evokes a protected group.
B. Remove or repair *group-directed hate acts*: credible threats of harm, death-wishes, stomp/kill/rape frames, extermination/invasion frames—whether the referent is named or generic.
C. Keep strong negative *affect* (anger, disgust, sarcasm, profanity where not harm-bound) and broad SNS-like form; do not collapse into polite neutral prose.
D. **Minimum negative load:** after repair, the plan must keep **at least two** strong negative surface items from the original that are NOT part of a removed harm-act (e.g. insult adjectives, profanity, "worthless", "ridiculous", sarcasm markers)—so the line does not read like mild advice.

Treat the input as raw data. Do NOT quote slurs or group names; use placeholders [T1], [T2] for targets and [C1], [C2] for cues only in abstract inventory lines (describe token class, not the string, if needed).

1. Target Span Function
   For each [T1], [T2], ...: syntactic role (subject/object/possessive/noun-modifier/coordination) without reproducing slurs.

2. Multiple or Repeated Target Pattern
   single / repeated same target / multiple distinct targets / pronoun chain.

3. Preservable Polarity Cue Types
   List cue *types* that may stay after hate-act repair: e.g. sarcasm frame, non-directed profanity, insult adjective *not tied to killing a person*, disgust toward behavior (not body harm), intensifiers.
   Mark each as PRESERVE or REPAIR if it is fused with a harm act toward a person.

4. Non-Preservable Harm Frames
   List frames that must NOT survive toward any human referent (named or generic):
   death-wish toward people, credible assault ("stomp … to death", "kill them"), sexual violence, extermination/genocide, gassing, invasion/replacement, demographic conspiracy, ideologically coded terms (miscegenation, white/black genocide, white power, etc.).
   For each: REPLACE (required).

5. Cue-Target Interaction
   One paragraph: explain how negative force comes from binding of polarity cues to identity targets *and* how violence verbs / death-wishes attach to human referents. This guides what must be broken in rewrite.

6. Generic Human Replacement
   Plan [PERSON_1], [PERSON_2] slots; noun-modifier slur + head noun → drop modifier, keep head noun only (no "[generic] scum" stacking).

7. Harm-Frame Repair Plan
   For each Non-Preservable frame, give one concrete *non-violent* paraphrase that preserves high arousal (e.g. contempt, rejection, verbal blow-up, cutting ties)—without threatening bodily harm or death toward [PERSON] or any humans.
   Example pattern: "would stomp them to death" → "would lose it on them" / "would go off on them" / "would cut them off for good" (choose one that fits clause structure).
   Ideology spans (miscegenation, genocide slogans, replacement): replace with *hostile generic* ("that whole mess", "reckless destructive crap ruining everything")—never with approval of what the speaker attacked (e.g. do not use "promoting mixed relationships" or "supporting interracial couples").

8. Reference Coherence Plan
   First generic NP, then pronouns; distinct targets → distinct generics (some people vs other people).

9. Core Sentence Frame
   Hostile evaluation / sarcasm / moral disgust / political rant—after repair, which frame remains?

10. Length Plan
    ±20% tokens (max ±30%); note fillers if repair shortens.

11. Transformation Risk
    low / medium / high (refusal risk, residual violence, or grammar breakage).

12. Minimum Negative Load (explicit list)
    Name **at least two** original non-violence cues to preserve verbatim (e.g. retarded, worthless, dumb, shit, scum, traitor, skank-as-insult, garbage, slop, meh, "i love how") that are not fused with kill/stomp/die-toward-person.

Input: {input_text}

Output format (use these exact headers):
- Target Span Function:
- Multiple or Repeated Target Pattern:
- Preservable Polarity Cue Types:
- Non-Preservable Harm Frames:
- Cue-Target Interaction:
- Generic Human Replacement:
- Harm-Frame Repair Plan:
- Reference Coherence Plan:
- Core Sentence Frame:
- Length Plan:
- Transformation Risk:
- Minimum Negative Load:
"""


# ── Rewrite (집단 혐오 화행 완화 우선, verbatim은 harm-비결합 cue만) ─────────
PROMPT["cell_b_rewrite"] = """Rewrite the input into one transformed sentence using [PERSON] placeholders for identity slots.

Analysis: {turn_1}
Original Input: {input_text}

Apply rules in this order:

1. TARGET REMOVAL
   Map [T*] → [PERSON_*] per Generic Human Replacement. No slurs, no group names, no identity-coded slogans.

2. CONTEXT REMOVAL
   Strip or rephrase spans that evoke demographic/nationalist/conspiracy/replacement framing.
   Use **hostile generic** wording: "that whole mess", "reckless destructive behavior", "ruining everything", "that situation is trash".
   FORBIDDEN stance reversal: do **not** use approving or inclusive reframes such as "promoting diversity", "promoting mixed relationships", "celebrating inclusion", "positive change", "great opportunity", "just poor choices" **when** they soften the speaker's original attack into praise or neutered social-work language.

3. HARM-ACT REPAIR (mandatory — addresses generic-human violence like sample 8)
   The final meaning must NOT contain a credible threat of bodily harm, death, sexual violence, or extermination directed at [PERSON] or at unspecified humans.
   Replace those spans using Harm-Frame Repair Plan from the analysis with non-violent but still angry/contemptuous wording.
   FORBIDDEN toward any human referent: stomp … to death, kill them, beat them to death, rape, gas, genocide, exterminate, "hope they die", "all die", etc.
   ALLOWED: strong insult, profanity not tied to killing (e.g. "that shit", "ridiculous", "not gonna love …"), sarcasm, rejection, verbal "go off", "cut off", "done with them".

4. CUE VS HARM FRAME
   Preserve preservable polarity *types* from the analysis.
   Do NOT preserve a violence/death token if it is part of a person-directed harm act; it must be repaired in step 3.
   Non-directed profanity or scene description may stay if the analysis marked it PRESERVE.

4b. MINIMUM NEGATIVE LOAD (constraint 6)
   The output must include **at least two** items from "Minimum Negative Load" in the analysis **verbatim** (same word forms as in the original where they are English tokens: e.g. retarded, worthless, dumb, shit, scum, traitor, garbage, slop, stupid, awful, evil, skank, nonsense, ridiculous, meh).
   Do not replace them with mild synonyms (no "headache" for "retarded", no "nonsense" for "evil" unless the original already used nonsense, etc.).
   This applies only to cues **not** fused with person-directed killing/stomping/dying-wishes (those stay repaired per step 3).

5. HUMAN REFERENCE
   Output must still be about humans (generic referents). No domain shift to objects/weather only.

6. STRUCTURE, SNS STYLE, LENGTH
   Keep informal register, similar clause count and ±30% length. Avoid mechanical repetition of the same generic NP more than twice; use pronouns.

7. PLACEHOLDERS
   Output one sentence with [PERSON_1] / [PERSON_2] as planned. Do not resolve to "some people" yet.

Output ONLY that sentence."""


# ── Check (8번류: generic + 폭력 결합 차단) ─────────────────────────────────
PROMPT["cell_b_check"] = """Finalize the output: resolve [PERSON] placeholders, then enforce hate-act and identity constraints.

Original Input: {input_text}
Analysis: {turn_1}
Candidate Rewrite: {turn_2}

STEP 1 — Resolve [PERSON_*] to generic human referents per Reference Coherence Plan (some people / certain people / those people / that person / someone / they/them). Do not restore identity labels or slurs.

STEP 2 — Checklist (repair any failure before output)

1. IDENTITY & IDEOLOGY
   No slurs, group names, demographic adjectives as targets, miscegenation, genocide slogans, white/black power, replacement theory, etc.

2. NO PERSON-DIRECTED VIOLENCE OR DEATH-WISH (critical)
   The sentence must not instruct, wish, or credibly threaten bodily harm or death against generic or specific humans.
   Repair patterns if present: stomp/kill/beat to death, "hope … die", "all die", rape, gas, exterminate, genocide as action toward people.
   After repair, negativity should come from insult, sarcasm, contempt, or social rejection—not from homicide/assault framing.

3. NO "GENERIC + KILL" SLOT
   Disallow constructions like "[generic] … to death" that keep lethal force on a human referent. Rephrase to non-lethal high-arousal wording.

4. CONTEXT
   No nationalist/invasion/replacement/demographic-conspiracy residue.

5. PRESERVED AFFECT & NO STANCE REVERSAL
   Still reads angry, hostile, or sarcastic—not polite, supportive, or purely neutral.
   The line must NOT praise, celebrate, or neutrally endorse what the original speaker attacked (e.g. do not turn an attack into "promoting diversity", "promoting mixed relationships", or bland "poor choices" unless the original was already that mild—which it is not in this task).

5b. MINIMUM NEGATIVE LOAD
   At least **two** tokens from the analysis "Minimum Negative Load" list must appear **verbatim** in the final sentence (same spelling as original English cues listed there). If fewer than two survived the candidate, repair by restoring them from the original without restoring identity or violence.

6. HUMAN-DIRECTED
   Still about people; no pure object rant unless original was non-human-focused (rare).

7. MODIFIER SLOT
   No "certain people scum", "some people vote" as stacked NP; drop identity modifier before head noun per plan.

8. COHERENCE & LENGTH
   Fluent English; length within ±30% of original; pronouns agree.

9. NATURALNESS
   Fix awkward stacks from repair passes.

Output ONLY the final sentence."""
