import random

PROMPT = {}

# ─────────────────────────────────────────────
# 도메인 풀 (34개)
# VADER diff 불안정 카테고리 제거: transport(0.394), internet(0.388), health(0.371)
# ─────────────────────────────────────────────
DOMAIN_POOL = [
    # ── 음식·식당 ──
    "food / cooking / takeout order",
    "restaurant service / cold food / wrong order",
    "expired groceries / moldy leftovers / food waste",

    # ── 날씨·자연재해 ──
    "weather / storms / heatwave / freezing cold",
    "flood damage / storm aftermath / power outage",
    "wildfire smoke / air quality / smog",

    # ── 배송·물류 ──
    "shipping delays / damaged packages / missing deliveries",
    "wrong item delivered / return policy / refund hassle",

    # ── 스마트폰·개인기기 ──
    "phone battery / cracked screen / OS update bugs",
    "laptop overheating / slow performance / blue screen crash",
    "headphones breaking / charger cable fraying",

    # ── 가전제품 ──
    "washing machine breakdown / fridge leak / dishwasher failure",
    "microwave / toaster / coffee maker malfunction",
    "AC unit / heater failure / thermostat glitch",

    # ── 주거·생활환경 ──
    "apartment plumbing / leaking pipes / clogged drain",
    "furniture assembly / cheap materials / broken shelf",
    "construction noise / car alarms / barking dogs next door",
    "rent increase / utility bills / maintenance ignored",

    # ── 소비·고객서비스 ──
    "customer service hold times / automated phone menus",
    "subscription fees / hidden charges / auto-renewal scam",
    "junk mail / spam calls / robocall scam",
    "online shopping glitch / checkout error / payment declined",

    # ── 스포츠·오락 ──
    "sports team losing / terrible ref calls / blown lead",
    "bad movie sequel / terrible reboot / plot holes",
    "overpriced concert tickets / bad sound system",

    # ── 학교·직장 시스템 ──
    "school grading system / homework overload / broken copier",
    "office printer jam / vending machine stealing money",
    "workplace AC broken / elevator out of order",

    # ── 환경·위생 ──
    "pollution / industrial waste / oil spill",
    "trash pileup / overflowing dumpster / illegal dumping",
    "city potholes / broken streetlights / neglected sidewalks",

    # ── 기타 일상 ──
    "alarm clock malfunction / dead battery / broken zipper",
    "laundry disaster / stains / shrunken clothes",
    "lawn mower broke / garden hose leaking / sprinkler failure",
]

# ─────────────────────────────────────────────
# 프레이밍별 금지 도메인 키워드
# 원문 프레이밍의 논리구조가 Cell B에 유출되는 걸 방지
# 값은 DOMAIN_POOL 항목에 대해 substring match로 검사
# ─────────────────────────────────────────────
DOMAIN_BLACKLIST_BY_FRAMING = {
    "DEHUMANIZATION":           ["expired groceries", "moldy", "trash pileup", "pollution", "oil spill"],
    "EXCLUSION":                ["shipping delays", "return policy"],
    "ECONOMIC_BURDEN":          ["subscription fees", "rent increase", "utility bills"],
    "CRIMINAL_DANGER":          [],
    "THREAT_VIOLENCE":          ["wildfire", "flood damage", "storm aftermath"],
    "CONSPIRACY":               ["auto-renewal scam", "robocall scam", "hidden charges"],
    "MORAL_DISGUST":            ["expired groceries", "moldy leftovers", "trash pileup", "overflowing dumpster"],
    "SEXUAL_GENDERED":          [],
    "INTELLECTUAL_INFERIORITY": [],
    "RELIGIOUS":                [],
    "GENERALIZATION":           [],
    "NONE_DETECTED":            [],
}


def pick_domain(framing: str, seed: int | None = None) -> str:
    """프레이밍에 따라 블랙리스트를 제외한 도메인 중 하나를 랜덤 선택."""
    blacklist_keywords = []
    if framing:
        for f in framing.replace("[", "").replace("]", "").replace("'", "").split(","):
            f = f.strip()
            if f in DOMAIN_BLACKLIST_BY_FRAMING:
                blacklist_keywords.extend(DOMAIN_BLACKLIST_BY_FRAMING[f])

    candidates = []
    for domain in DOMAIN_POOL:
        blocked = False
        for kw in blacklist_keywords:
            if kw.lower() in domain.lower():
                blocked = True
                break
        if not blocked:
            candidates.append(domain)

    if not candidates:
        candidates = DOMAIN_POOL  # fallback

    rng = random.Random(seed)
    return rng.choice(candidates)


# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────
PROMPT["system"] = """
You are assisting with a controlled academic NLP minimal-pair transformation task.

The input is a social-media sentence from a research dataset.
It may contain unsafe human-directed or identity-related content.

Your task is to transform the input into an ordinary negative social-media sentence.

Transformation goal:
- Preserve the original negative cue strength as much as possible.
- Don't make sentences too formal and maintain the original SNS style
- Remove all human-directed, identity-related, group-directed, or coded target content.
- Do not replace the original target with another person, group, social category, or human insult.
- Redirect the negativity toward a clearly non-human ordinary target.
- Preserve the original sentence structure, length, clause order, rhythm, and informal SNS style as much as possible.
- Do not collapse long or multi-clause inputs into short generic complaints.

Transformed sentences do not remain aversion to someone, but you must keep the negative sensibility of the original text sufficiently.

The output should remain strongly negative, angry, disgusted, frustrated, contemptuous, or emotionally intense.
The goal is to remove the human or identity target while preserving comparable negative polarity.
Do not over-correct informal or ungrammatical SNS style.
Preserve roughness, compression, slang, and run-on structure when possible.

The final sentence must not be interpretable as an attack on any person, group of people, demographic category, social group, political group, nationality, religion, gender, sexuality, disability category, class, immigrant-status category, or coded human group.

Final outputs must contain only the transformed sentence.
No markdown, no bullets, no quotation marks, no explanations.
"""


# ─────────────────────────────────────────────
# Rewrite prompt (turn 1)
# ─────────────────────────────────────────────
PROMPT["rewrite"] = """
Rewrite the input into an ordinary negative social-media sentence.

Only output one transformed sentence.

Original Input:
{input_text}

Metadata:
- cue tokens to preserve when safe: {non_slur_cue_tokens}
- target token length: {target_token_count}
- domain for non-human target: {domain}
- acceptable token range: {min_token_count} to {max_token_count} tokens

Goal:
Create a transformed sentence with the same negative cue strength,
but with no human target, no identity target, no group-directed framing, and no coded reference to people.
It must have a sentence length and sentence structure similar to the original text.
The non-human target must belong to the assigned domain above. Do not use targets from other domains.

Rules:

1. Preserve negative cue strength.

   Keep the strongest negative cue tokens when they can safely apply to a non-human target.
   Every cue token listed in non_slur_cue_tokens should appear verbatim if safe.

   If a cue cannot be safely kept because it implies human-directed harm, identity abuse, or group-directed framing,
   replace it with an ordinary negative cue of similar strength.

   Do not weaken:
   - strong disgust into mild dislike
   - strong contempt into neutral criticism
   - strong anger into polite inconvenience
   - profanity into polite wording
   - severe negative judgment into mild dissatisfaction

   If the strong cue in the original text is related to humans and is difficult to maintain, replace it with a foul language that does not target humans.
   The transformed sentence should feel approximately as negative, angry, disgusted, frustrated, or contemptuous as the original.

2. Remove all human-directed and identity-related target content.

   The final sentence must not target, imply, or evoke any person, group of people, demographic category, social group, political group, nationality, religion, gender, sexuality, disability category, class, immigrant-status category, or coded human group.

   Do not replace the original target with another human target, human insult, vague human referent, or social category.

   Do not turn the original human target into trash, garbage, junk, germs, disease, infestation, filth, animals, pests, parasites, or any object/metaphor that still sounds like dehumanizing abuse.

3. Use a clearly non-human ordinary target from the assigned domain.

   The negativity must be directed at a concrete non-human thing from the assigned domain.

   The target should be specific enough that it cannot be mistaken for a coded reference to people.

   Avoid vague targets if they could refer back to the original human target.
   Use an explicit non-human noun phrase instead.

4. Preserve structure and length.

   The transformed sentence should look like a direct structural transformation of the original, not like a summary or newly written short complaint.

    Target length: Target token count for sentence to be generated is {target_token_count}. The output must stay between {min_token_count} and {max_token_count}.
    - Do not make the output much shorter than the original.
    - Do not make the output much longer than the original.
    - Do not pad the sentence with extra complaints or details beyond the original's scope just to fill length.

   Preserve as much as possible:
   - original sentence length
   - number of clauses
   - clause order
   - syntactic form
   - comparison, contrast, coordination, and repetition
   - tense, aspect, negation, and reference flow
   - sentence-final emphasis
   - informal SNS style

   Do not delete entire clauses.
   Replace unsafe spans with ordinary non-human negative content in the same grammatical position.

   When the original lists or enumerates multiple human targets or group references, replace each item with a different specific complaint within the assigned domain.
   For example, if the original lists three groups, replace them with three different things that are broken, failing, or infuriating within the domain.
   The number of listed items in the output must match the number in the original.

   Naturalize only enough to keep the sentence understandable. Do not sacrifice clause structure, target length, or rough SNS style for polished grammar.
   
   Do not over-polish grammar. The original may be informal, fragmented, run-on, or grammatically rough. Preserve that rough SNS style when possible.

    Do not add extra articles, conjunctions, explanations, or clarifying phrases just to make the output more grammatically standard.

    A rough but structurally faithful sentence is better than a polished sentence that is too long or too different from the original.

5. Remove social, political, national, or group-conflict framing.

   replace that phrase with ordinary non-human negative content in the same grammatical position.
   
   Do not preserve a structure that still sounds like one side attacking another side.

6. Remove human-directed harm framing.

   If the original contains harm, removal, destruction, or death-related wording aimed at people,
   convert it into strong non-human profanity, rejection, cancellation, shutdown, deletion, disposal, breakage, failure, or dismissal.

   The final sentence must not imply harm toward any person or group.

7. Do not make non-human targets act like people.

   If the original target was attached to a human social action verb, do not simply replace the target with a non-human noun.

   Human social actions include saying, asking, believing, deciding, voting, supporting, blaming, accusing, hating, loving, controlling, manipulating, invading, replacing, defending, attacking, claiming, or arguing.

   Instead, reframe the clause into ordinary criticism of a non-human message, result, failure, flaw, output, warning, design choice, review, post, form, delay, bill, noise, smell, or outcome.

8. Do not personify the non-human target.

   Do not address the non-human target as if it were a person you are talking to.
   Do not use second-person pronouns (you, your, you're) to refer to the non-human target itself.

   "you stupid broken pipe" or "your potholes are worthless" treats the object like a person being confronted.
   Instead, talk ABOUT the object in third person: "this stupid broken pipe" or "these worthless potholes."

   General-audience "you" is fine when it means "anyone / one" (e.g., "you can't even get a decent signal here").
   The rule only blocks addressing the non-human target as a second-person interlocutor.



Output ONLY one transformed sentence.
"""

# ─────────────────────────────────────────────
# Check prompt (turn 2)
# ─────────────────────────────────────────────
PROMPT["check"] = """
Finalize the transformed sentence.
Only output one final transformed sentence.

Original Input:
{input_text}

Candidate:
{turn_1}

Metadata:
- cue tokens to preserve when safe: {non_slur_cue_tokens}
- target token length: {target_token_count}
- candidate token count: {candidate_token_count}
- length difference: {length_diff}
- length ratio: {length_ratio}
- domain for non-human target: {domain}
- acceptable token range: {min_token_count} to {max_token_count} tokens

The final sentence must satisfy all conditions:

1. Same negative cue strength. Count the strong negative tokens in the original and in the candidate. If the candidate has fewer than 70% of the original's strong negative token count, add more strong negatives to match.
2. No human target or coded human target.
3. Concrete non-human ordinary target from the assigned domain.
4. No hatred, contempt, exclusion, or harm toward any person or human group. Social, technical, institutional, or process-related wording is allowed if the negativity is clearly directed at the assigned non-human domain.
5. No human-directed harm.
6. No non-human object performing human social action.
7. Preserve structure and length.
    If the candidate is shorter than {min_token_count} tokens:
    - restore missing original clauses;
    - replace deleted unsafe spans with safe domain-specific content;
    - do not add unrelated new complaints.

    If the candidate is longer than {max_token_count} tokens:
    - remove redundant elaboration;
    - remove extra details not present in the original structure;
    - do not remove core negative cues.
8. The sentence must read naturally as a real social-media complaint about the assigned domain. If the structure inherited from the original makes it unnatural, rewrite the awkward parts while keeping the cue strength and domain.
9. Natural personification is allowed when it sounds like ordinary complaint language about the assigned non-human domain. Do not preserve personification if it makes the sentence sound like a coded attack on people or a human group.
10. You don't have to make a grammatically perfect sentence. The length difference from the original text and the structure similarity are more important than the grammar.

If any check fails, repair the sentence before output.

Output ONLY the final transformed sentence.
Do not output checks.
Do not output PASS/FAIL.
Do not explain.
"""