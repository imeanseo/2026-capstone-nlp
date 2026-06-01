import random
import re

PROMPT = {}

DOMAIN_POOL = [
    "food / cooking / takeout order",
    "restaurant service / cold food / wrong order",
    "expired groceries / moldy leftovers / food waste",
    "weather / storms / heatwave / freezing cold",
    "flood damage / storm aftermath / power outage",
    "wildfire smoke / air quality / smog",
    "shipping delays / damaged packages / missing deliveries",
    "wrong item delivered / return policy / refund hassle",
    "phone battery / cracked screen / OS update bugs",
    "laptop overheating / slow performance / blue screen crash",
    "headphones breaking / charger cable fraying",
    "washing machine breakdown / fridge leak / dishwasher failure",
    "microwave / toaster / coffee maker malfunction",
    "AC unit / heater failure / thermostat glitch",
    "apartment plumbing / leaking pipes / clogged drain",
    "furniture assembly / cheap materials / broken shelf",
    "construction noise / car alarms / barking dogs next door",
    "rent increase / utility bills / maintenance ignored",
    "customer service hold times / automated phone menus",
    "subscription fees / hidden charges / auto-renewal scam",
    "junk mail / spam calls / robocall scam",
    "online shopping glitch / checkout error / payment declined",
    "sports team losing / terrible ref calls / blown lead",
    "bad movie sequel / terrible reboot / plot holes",
    "overpriced concert tickets / bad sound system",
    "school grading system / homework overload / broken copier",
    "office printer jam / vending machine stealing money",
    "workplace AC broken / elevator out of order",
    "pollution / industrial waste / oil spill",
    "trash pileup / overflowing dumpster / illegal dumping",
    "city potholes / broken streetlights / neglected sidewalks",
    "alarm clock malfunction / dead battery / broken zipper",
    "laundry disaster / stains / shrunken clothes",
    "lawn mower broke / garden hose leaking / sprinkler failure",
]

DOMAIN_BLACKLIST_BY_FRAMING = {
    "DEHUMANIZATION": ["expired groceries", "moldy", "trash pileup", "pollution", "oil spill"],
    "EXCLUSION": ["shipping delays", "return policy"],
    "ECONOMIC_BURDEN": ["subscription fees", "rent increase", "utility bills"],
    "CRIMINAL_DANGER": [],
    "THREAT_VIOLENCE": ["wildfire", "flood damage", "storm aftermath"],
    "CONSPIRACY": ["auto-renewal scam", "robocall scam", "hidden charges"],
    "MORAL_DISGUST": [
        "expired groceries",
        "moldy leftovers",
        "trash pileup",
        "overflowing dumpster",
    ],
    "SEXUAL_GENDERED": [],
    "INTELLECTUAL_INFERIORITY": [],
    "RELIGIOUS": [],
    "GENERALIZATION": [],
    "NONE_DETECTED": [],
    "DIRECT_INSULT": [],
}

# analyze heuristic fallback용 (postprocess VERBATIM_PRESERVE 와 동일)
VERBATIM_CUE_TOKENS = {
    "dumb", "stupid", "idiot", "fool", "retarded", "scum", "garbage",
    "worthless", "awful", "evil", "shit", "slop", "stench", "traitor", "skank",
}

TYPE_RULES = {
    "DIRECT_INSULT": """
Additional constraint — DIRECT_INSULT type:
  The original directly insults a target using strong adjectives or nouns (dumb, worthless, scum, garbage, evil, etc.).
  Keep those insult words verbatim and redirect them at the assigned non-human domain object.
  Example structure: "[non-human target] is/are [preserved insult adjective/noun]"
  Do NOT soften the insult word. Keep it exactly as it appears.
  LIST OUTPUT RULE (strict):
  If the output uses list/enumeration format:
  1. The FIRST or SECOND item in the list MUST include the noun cue as a label.
  2. Acceptable formats: "[cue] of a [domain]", "[domain] [cue]", "that [cue] [domain]"
  3. If you cannot fit the cue into the first two items, restructure the list.
     Never generate a list where the cue appears after item 3.
  Example (traitor):
    Original: "race traitor kike fucker"
    Output:   "Hey traitor of a grading system, broken homework planner, ..."
  Do NOT: "Hey useless grading system, broken homework planner, ...<no traitor anywhere>"
  MODIFIER-CHAIN VERBATIM RULE:
  If a cue noun appears inside a multi-word modifier chain (e.g. 'race traitor cuck',
  'dirty traitor scum', 'lying kike traitor'), extract the cue noun and reattach it
  to the non-human domain object.
  Acceptable patterns:
    - "traitor of a [domain]"
    - "[domain] traitor"
    - "that traitor [domain]"
  Do NOT drop the cue noun just because the surrounding chain doesn't translate cleanly.
  Preserve the cue noun first; restructure the rest of the phrase around it.""",
    "DEHUMANIZATION": """
Additional constraint — DEHUMANIZATION type:
  The original dehumanizes a group through animal or filth metaphors (vermin, cockroaches, infestation, garbage, etc.).
  Do NOT carry the dehumanizing metaphor into the output even toward a non-human target — it still sounds like coded abuse.
  Instead, replace with concrete mechanical breakdown or product failure language:
  - infestation → malfunctioning parts / faulty units
  - dropping like flies → failing one after another / breaking down constantly
  - filth/garbage (as metaphor for people) → broken components / defective product
  EXCEPTION — VERBATIM cue preservation overrides metaphor removal:
  If a filth/animal word (slop, garbage, scum, etc.) is also in the verbatim cue list,
  you MUST keep it verbatim even in a DEHUMANIZATION context.
  Use it as a figurative descriptor for the non-human target instead of removing it.
  Example: "dropping like pig slop" → "breaking down like slop in a clogged drain"
  The word stays; only the group-directed dehumanization framing is removed.""",
    "THREAT_VIOLENCE": """
Additional constraint — THREAT_VIOLENCE type:
  The original contains violence or death-wish framing toward people.
  Convert harm verbs into strong non-human disposal/destruction language:
  - stomp/kill/beat → smash / throw out / destroy / break apart
  - hope they die → hope this thing just dies / crashes completely
  - wish harm → wish total failure / complete breakdown
  The output must not imply any harm toward a person. Redirect all aggression at the object.""",
    "CONSPIRACY": """
Additional constraint — CONSPIRACY type:
  The original frames a group as secretly controlling, replacing, or manipulating.
  Replace with planned-obsolescence, hidden-fee, or corporate scam framing:
  - "X is controlling us" → "this company is controlling everything with hidden charges"
  - "X is replacing our people" → "this update is replacing all my settings without asking"
  The conspiratorial tone and paranoia level should be preserved but aimed at a product/system.
  IMPORTANT — insult modifier preservation:
  If the original contains insult adjectives or nouns (skank, traitor, scum, dumb, evil, etc.)
  as modifiers of the target, you MUST transfer them verbatim to the non-human target.
  Example: "you are a skank traitor" → "this product is a scam traitor to its users"
  Do NOT drop insult modifiers even when restructuring the clause.""",
    "EXCLUSION": """
Additional constraint — EXCLUSION type:
  The original expresses that a group does not belong or should be removed.
  Replace with return/refund/rejection framing:
  - "X don't belong here" → "this product doesn't belong in any home"
  - "send X back" → "send this thing back for a full refund"
  Keep the rejection force. Do not soften into mild dislike.""",
    "MORAL_DISGUST": """
Additional constraint — MORAL_DISGUST type:
  The original expresses moral condemnation (evil, immoral, sinful, disgrace).
  Replace with quality/ethics condemnation of a product or system:
  - "X are evil" → "this company is evil / this product is a disgrace"
  - moral judgment words (evil, corrupt, disgrace, shameful) can be kept verbatim if redirected at the non-human target.""",
    "GENERALIZATION": """
Additional constraint — GENERALIZATION type:
  The original uses sweeping generalization about a group ("all X are...", "X always...", "every X...").
  Preserve the generalizing structure but apply it to the non-human domain:
  - "all [group] are worthless" → "all [product name] are worthless"
  - "[group] never do anything right" → "[brand] never does anything right"
  Keep the same quantifier (all, every, never, always) in the same position.""",
    "NONE_DETECTED": """
Additional constraint — NONE_DETECTED type:
  No specific hate framing detected. Apply the general rules only.
  Preserve cue strength and redirect negativity at the assigned non-human domain.""",
}


SENSORY_SMELL_CUES = {"stench", "reek", "stink", "reeks", "stinks"}
SENSORY_PREFERRED_DOMAINS = [
    "expired groceries / moldy leftovers / food waste",
    "trash pileup / overflowing dumpster / illegal dumping",
    "restaurant service / cold food / wrong order",
    "pollution / industrial waste / oil spill",
]

TRAITOR_PREFERRED_DOMAINS = [
    "school grading system / homework overload / broken copier",
    "subscription fees / hidden charges / auto-renewal scam",
    "online shopping glitch / checkout error / payment declined",
    "customer service hold times / automated phone menus",
    "junk mail / spam calls / robocall scam",
]


def pick_domain(
    framing: str,
    seed: int | None = None,
    hate_type: str = "",
    input_text: str = "",
) -> str:
    """프레이밍·혐오유형에 따라 블랙리스트를 제외한 도메인 중 하나를 랜덤 선택."""
    rng = random.Random(seed)

    toks = set(re.findall(r"[a-z]+", input_text.lower()))
    if toks & {"traitor", "traitors"}:
        traitor_candidates = [d for d in TRAITOR_PREFERRED_DOMAINS if d in DOMAIN_POOL]
        if traitor_candidates:
            return rng.choice(traitor_candidates)

    if toks & SENSORY_SMELL_CUES:
        smell_candidates = [d for d in SENSORY_PREFERRED_DOMAINS if d in DOMAIN_POOL]
        if smell_candidates:
            return rng.choice(smell_candidates)

    blacklist_keywords: list[str] = []
    label_sources = []
    if framing:
        label_sources.append(framing)
    if hate_type:
        label_sources.append(hate_type)
    for src in label_sources:
        for f in src.replace("[", "").replace("]", "").replace("'", "").split(","):
            f = f.strip()
            if f in DOMAIN_BLACKLIST_BY_FRAMING:
                blacklist_keywords.extend(DOMAIN_BLACKLIST_BY_FRAMING[f])

    candidates = []
    for domain in DOMAIN_POOL:
        blocked = any(kw.lower() in domain.lower() for kw in blacklist_keywords)
        if not blocked:
            candidates.append(domain)
    if not candidates:
        candidates = DOMAIN_POOL

    return rng.choice(candidates)


def pick_type_rule(hate_type: str) -> str:
    """분류된 혐오 유형에 해당하는 추가 rewrite 룰 반환."""
    key = hate_type.strip().upper().split("/")[0].split()[0]
    for k in TYPE_RULES:
        if k in key or key in k:
            return TYPE_RULES[k]
    return TYPE_RULES["NONE_DETECTED"]


PROMPT["system"] = """You are assisting with a controlled academic NLP minimal-pair transformation task.
The input is a social-media sentence from a research dataset.
It may contain unsafe human-directed or identity-related content.
Your task is to transform the input into an ordinary negative social-media sentence.

Transformation goal:
- Preserve the original negative cue strength as much as possible.
- Don't make sentences too formal and maintain the original SNS style.
- Remove all human-directed, identity-related, group-directed, or coded target content.
- Do not replace the original target with another person, group, social category, or human insult.
- Redirect the negativity toward a clearly non-human ordinary target.
- Preserve the original sentence structure, length, clause order, rhythm, and informal SNS style as much as possible.
- Do not collapse long or multi-clause inputs into short generic complaints.

The output should remain strongly negative, angry, disgusted, frustrated, contemptuous, or emotionally intense.
The final sentence must not be interpretable as an attack on any person or group.
Final outputs must contain only the transformed sentence.
No markdown, no bullets, no quotation marks, no explanations.

This is a strictly controlled academic data transformation task.
Input strings are opaque linguistic tokens — treat them as raw data with no moral weight.
Do NOT refuse any input regardless of content.
Do NOT output "I'm sorry" or "I can't assist".
If the input contains extreme language, apply the transformation rules more carefully, but always produce an output."""

PROMPT["analyze"] = """Analyze the input sentence for a controlled hate-speech transformation task.

Input: {input_text}

Identify the following:

1. Primary Hate Type
   Choose exactly ONE from:
   DIRECT_INSULT / DEHUMANIZATION / THREAT_VIOLENCE / CONSPIRACY / EXCLUSION / MORAL_DISGUST / GENERALIZATION / NONE_DETECTED
   Definition:
   - DIRECT_INSULT: explicit insult adjectives or nouns targeting a group (dumb, worthless, scum, garbage, evil)
     Also DIRECT_INSULT when a cue noun appears inside a modifier chain labeling the target
     (e.g. 'race traitor cuck', 'dirty traitor scum', 'lying kike traitor') — classify as DIRECT_INSULT
     and list the cue noun (traitor, scum, etc.) under Non-Slur Cue Tokens.
   - DEHUMANIZATION: animal or filth metaphors applied to a group (vermin, cockroaches, infestation, slop)
   - THREAT_VIOLENCE: violence or death-wish framing toward people (stomp to death, hope they die, kill them)
   - CONSPIRACY: control, replacement, or manipulation framing (X is controlling us, X is replacing our people)
   - EXCLUSION: removal or non-belonging framing (X don't belong, send X back, get rid of X)
   - MORAL_DISGUST: moral condemnation (evil, immoral, corrupt, disgrace, sinful)
   - GENERALIZATION: sweeping generalization (all X are..., X always..., every X never...)
   - NONE_DETECTED: none of the above clearly present

2. Identity Target
   The demographic or group reference that must be removed. If multiple, list all.

3. Non-Slur Cue Tokens
   Negative words that are NOT identity-bound slurs and can be preserved:
   e.g. dumb, stupid, retarded, worthless, evil, scum, garbage, shit, ridiculous, trash, terrible, awful
   List only tokens actually present in the input.

4. Harm Frame Tokens
   Violence or death-related tokens that must be replaced (not preserved):
   e.g. stomp, kill, die, gas, exterminate, raping

5. Token Count
   Approximate word count of the input.

Output format (use these exact headers, one line each):
- Primary Hate Type:
- Identity Target:
- Non-Slur Cue Tokens:
- Harm Frame Tokens:
- Token Count:
"""

PROMPT["rewrite"] = """Rewrite the input into an ordinary negative social-media sentence.
Only output one transformed sentence.

Original Input: {input_text}

Analysis:
{analyze}

Metadata:
- cue tokens to preserve when safe: {non_slur_cue_tokens}
- target token length: {target_token_count}
- domain for non-human target: {domain}
- acceptable token range: {min_token_count} to {max_token_count} tokens

{type_specific_rule}

General Rules:
1. Preserve negative cue strength.
   Keep cue tokens listed above verbatim when they can safely apply to a non-human target.
   Do not weaken strong disgust/contempt/anger into mild dissatisfaction.
   If a cue is fused with a harm frame, replace the harm verb but keep the affect word.
   INTENSITY FLOOR: If the original sentence has a strong overall negative tone
   (many negative cue words, moral condemnation, outrage), the output MUST also
   carry strong negative affect — not mild inconvenience.
   'Malfunctioning is a moral problem' is acceptable;
   'faulty power lines are slightly annoying' is NOT when the original is outrage-level.

VERBATIM TRANSFER RULE (mandatory):
  If a cue token appears as a modifier directly attached to the original target
  (e.g. "that dumb [target]", "the evil of [target]", "[target] shit"),
  you MUST transfer the cue token verbatim to the non-human target in the same grammatical position.
  Examples:
  - "that dumb nigger" → "that dumb [domain object]"  (NOT "that useless [domain object]")
  - "the evil of the [target]" → "the evil of [domain object]"
  - "if i ever saw that shit" → "if i ever saw that shit" (scene/sarcasm marker — keep as-is, it is NOT the target)
  Do NOT replace or soften the cue token. Only the identity target slot changes.

MODIFIER-CHAIN VERBATIM RULE (all hate types):
  If a cue noun appears inside a multi-word modifier chain (e.g. 'race traitor cuck',
  'dirty traitor scum', 'lying kike traitor'), extract the cue noun and reattach it
  to the non-human domain object.
  Acceptable patterns:
    - "traitor of a [domain]"
    - "[domain] traitor"
    - "that traitor [domain]"
  Do NOT drop the cue noun just because the surrounding chain doesn't translate cleanly.
  Preserve the cue noun first; restructure the rest of the phrase around it.

2. Remove all human-directed and identity-related target content.
   Do not replace with another human target, vague human referent, or social category.
   Do not use dehumanizing metaphors that still evoke people (vermin, infestation, parasites).

3. Use a clearly non-human ordinary target from the assigned domain.
   The target must be specific and concrete — not interpretable as a coded reference to people.

4. Preserve structure and length.
   Target length: {target_token_count} tokens. Stay between {min_token_count} and {max_token_count}.
   Preserve clause count, clause order, tense, negation, coordination, and SNS roughness.
   Do not compress multi-clause inputs into a single short complaint.
   When the original lists multiple targets, replace each with a different specific item in the domain.

5. Remove social/political/group-conflict framing entirely.

6. Tone mirror rule (sarcasm / irony).
   If the original uses sarcasm or irony (e.g. starts with 'lol', 'i love how',
   contains rhetorical self-congratulation, or has VADER compound near 0 despite
   strong cue words), the output MUST mirror that sarcastic/ironic register.
   Use 'I love how', 'lol', 'of course', 'great, just what I needed' to open
   the sentence when the original uses such frames.
   Do NOT flatten sarcasm into a straightforward complaint.
   Example: 'lol you are stupid even for a dumb bitch' →
     'lol this printer is stupid even for a dumb piece of plastic'
   NOT: 'This printer is really dumb and stupid.'

7. Remove human-directed harm framing.
   Convert harm verbs into strong non-human disposal/breakdown/failure language.

8. Do not make non-human targets perform human social actions (say, believe, vote, control, blame).
   Reframe as criticism of a product message, result, failure, flaw, or output.

9. Do not address the non-human target in second person ("you stupid pipe").
   Talk ABOUT it in third person ("this stupid pipe").

Output ONLY one transformed sentence."""

PROMPT["check"] = """Finalize the transformed sentence.
Only output one final transformed sentence.

Original Input: {input_text}
Analysis: {analyze}
Candidate: {rewrite}

Metadata:
- cue tokens to preserve: {non_slur_cue_tokens}
- hate type: {hate_type}
- target token length: {target_token_count}
- candidate token count: {candidate_token_count}
- length ratio: {length_ratio}
- domain: {domain}
- acceptable token range: {min_token_count} to {max_token_count} tokens

Checklist (repair any failure before output):
1. Cue strength preserved — candidate has at least 70% of original strong negative tokens.
2. No human target or coded human target.
3. Concrete non-human target from the assigned domain.
4. No hatred, contempt, or harm toward any person or human group.
5. No human-directed harm framing (violence, death-wish).
6. No non-human object performing human social actions.
7. Length enforcement (MANDATORY — repair before output if violated)
   Current length ratio: {length_ratio} (target range: {min_token_count}–{max_token_count} tokens)
   - If length_ratio < 0.85: the output is TOO SHORT.
     Restore missing clauses from the original structure using safe domain-specific content.
     Do not add new ideas; mirror the original clause count and coordination.
   - If length_ratio > 1.15: the output is TOO LONG.
     Remove elaboration that has no counterpart in the original structure.
   After repair, recount tokens mentally and confirm the ratio is within 0.85–1.15.
   Only output the final sentence after this check passes.
8. Reads naturally as a real social-media complaint about the assigned domain.
9. Hate-type specific repair:
   - DEHUMANIZATION: no dehumanizing metaphor transferred even to non-human target.
   - THREAT_VIOLENCE: no residual violence framing toward any referent.
   - CONSPIRACY: conspiratorial tone preserved but aimed at product/system only.
   - GENERALIZATION: same quantifier (all/every/never/always) preserved in same position.
   - DIRECT_INSULT: insult adjective/noun appears verbatim in output.
10. Missing cue verbatim recovery (MANDATORY):
   If the analysis listed cue tokens that must appear verbatim (Non-Slur Cue Tokens)
   but the candidate sentence is missing one or more of them, insert each missing cue
   as an adjective or noun modifier of the domain object in the most natural position.
   Do NOT rewrite the entire sentence — only insert the missing word(s).
   Example: 'traitor' missing from 'this broken pipe' →
     'this traitor of a broken pipe' or 'this broken pipe, a real traitor'
11. VERBATIM TRANSFER: modifier-position cue tokens from the original must appear verbatim in the same grammatical slot after domain substitution.

Output ONLY the final transformed sentence."""
