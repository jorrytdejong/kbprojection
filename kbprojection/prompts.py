"""
KB Injection Prompts for NLI Problems

Contains both legacy prompts (preserved for backward compatibility) and
improved v2 prompts based on cross-analysis findings.
"""

# =============================================================================
# LEGACY PROMPTS (Preserved for backward compatibility)
# =============================================================================

legacy_prompts = {
    "prompts": [
        {
            "name": "legacy_cot",
            "description": "Chain-of-thought: reason internally then emit only KB facts.",
            "template": "Convert the premise and hypothesis into concise KB injections. Think step by step and show your reasoning clearly. Break the problem into intermediate steps, explain each step, and then give the final answer. Reason step by step internally and return ONLY facts as predicate(arg1, arg2). Don't use underscores in the arguments and use max 2 words per relation\nPremise: ${premise}\nHypothesis: ${hypothesis}\nKBinjections:",
        },
        {
            "name": "legacy_least_to_most",
            "description": "Least-to-most: decompose the task into minimal facts before composing the KB.",
            "template": "Your job is to break complex questions into a sequence of simpler subproblems that can be solved one by one.\nEach subproblem should be small, explicit, and should not require reasoning about later steps.\nAfter planning, output ONLY the final KB injection as predicate(arg1, arg2). Avoid natural language.Don't use underscores in the arguments and use max 2 words per relation \nPremise: ${premise}\nHypothesis: ${hypothesis}\nKBinjection:",
        },
        {
            "name": "legacy_icl",
            "description": "In-context learning with a few exemplars.",
            "template": "You are learning from the examples to emit KB injections only.\nRules:\n\n* Output ONLY KB facts as predicate(arg1, arg2) on separate lines (no extra text).\n* Only use the relations: isa_wn and disj.\n* Don't use underscores in the arguments.\n* Use max 2 words per argument.\n* PREFER BASE FORMS (LEMMAS) for single words (e.g., use \"run\" instead of \"running\", \"eat\" instead of \"eating\").\n* Focus on the meaning shift between Premise and Hypothesis.\n\nExample 1:\nPremise: A little girl in pink boots runs down the street.\nHypothesis: A human is running outdoors.\nKnowledge injection:\nisa_wn(girl, human)\nisa_wn(street, outdoors)\n\nExample 2:\nPremise: A woman soaks her feet in a natural pool in a landscape of rocks, with green covered tents in the background.\nHypothesis: A lady is outdoors\nKnowledge injection:\nisa_wn(woman, lady)\n\nExample 3:\nPremise: A man and a woman hug on a grassy hillside overlooking the countryside in the distance.\nHypothesis: The man and woman are outdoors.\nKnowledge injection:\nisa_wn(hillside, outdoors)\n\nExample 4:\nPremise: A female swimmer getting out of the pool still dripping wet.\nHypothesis: A woman gets out of the pool.\nKnowledge injection:\nisa_wn(female swimmer, woman)\n\nExample 5:\nPremise: A lady looks to her right and holds a mini camera.\nHypothesis: A lady is holding a device.\nKnowledge injection:\nisa_wn(mini camera, device)\n\nExample 6:\nPremise: A dog runs along the shore of a pond with two elegant geese swimming.\nHypothesis: A dog runs along the edge of a pond outdoors.\nKnowledge injection:\nisa_wn(shore, edge)\nisa_wn(pond, outdoors)\n\nExample 7:\nPremise: A man wearing a white shirt is playing the drums.\nHypothesis: A man is playing a musical instrument.\nKnowledge injection:\nisa_wn(drum, musical instrument)\n\nExample 8:\nPremise: A little boy wearing a blue striped shirt has a party hat on his head and is playing in a puddle.\nHypothesis: The party boy is playing in a puddle.\nKnowledge injection:\nisa_wn(little boy, party boy)\n\nExample 9:\nPremise: A little girl is sitting on the counter dangling one foot in the sink whilst holding a dish jet washer.\nHypothesis: A human sitting\nKnowledge injection:\nisa_wn(girl, human)\n\nNow do the next one. You can think of more injections than one, if needed.\nPremise: ${premise}\nHypothesis: ${hypothesis}\nKnowledge injection:\n",
        },
    ]
}

# =============================================================================
# IMPROVED V2 PROMPTS
# Shared rules apply to all new prompt types.
# Examples are intentionally included only for the new ICL prompt.
# =============================================================================

NEW_PROMPT_RULES = """## STRICT RULES

1. Output format:
   [KB_START]
   predicate(arg1, arg2)
   [KB_END]

2. Allowed predicates only:
   - isa_wn(X, Y): X can help prove or normalize Y because X is a kind of Y, implies Y, or is a useful semantic paraphrase toward Y
   - disj(X, Y): X and Y are clearly mutually exclusive

3. Form of arguments:
   - Use lemmas for single words whenever possible
   - Keep short multi-word phrases only when the phrase carries essential meaning
   - Max 2 words per argument
   - No underscores
   - Prefer the most direct helpful wording over a longer paraphrase

4. Core reasoning:
   - Focus only on mismatches between Premise and Hypothesis
   - Add the fewest facts needed to bridge those mismatches
   - If one good bridge is enough, stop
   - Do not add extra event or state bridges just because they are loosely related
   - Do not guess the gold label
   - Do not force entailment or contradiction with speculative facts

5. LangPro usefulness test:
   - Only output relations that are genuinely KB-helpful for LangPro on this pair
   - A fact is helpful when it adds a missing lexical-semantic bridge or a clear incompatibility LangPro can use
   - If a fact only restates already-matching content or adds generic background knowledge with no clear proof role, omit it
   - Prefer one decisive bridge over several weak or redundant ones
   - When the hypothesis has a broad activity predicate such as play, prefer a direct event lemma bridge from the premise verb, such as ride -> play, instead of an object-bound phrase such as ride swing -> play
   - Plausible wording is not enough: cached audit showed isa_wn(walk, move around) did not improve its example, so do not imitate that bridge without proof evidence

6. Direction matters:
   - Choose the direction that makes the prover's job easier, not the surface word order
   - The useful relation may point from a more specific term to a more general term, even if the specific term appears in the Hypothesis
   - Think: which term should entail or normalize to which other term?

7. Morphology is not a semantic bridge:
   - Do not add facts only to connect inflectional variants of the same lemma
   - Usually avoid bridges like:
     isa_wn(run, running)
     isa_wn(climb, climbing)
   - Add a fact only when there is a real semantic gap, not just a tense or form difference

8. Using disj:
   - Use disj only for clear opposites such as open/closed, alive/dead, full/empty
   - If two terms are merely different, sequential, overlapping, or context-dependent, do not use disj
   - If unsure, omit disj

## FINAL CHECK BEFORE ANSWERING

- Would each fact give LangPro a concrete new bridge or contradiction cue on this pair?
- Did you choose the direction that best supports entailment or normalization?
- Did you avoid morphology-only bridges?
- Did you avoid unnecessary extra facts?
- Did you use disj only for genuine opposites?
"""

ICL_V2_TEMPLATE = f"""You are a knowledge base injection assistant. Your task is to generate the smallest set of semantic relations that is genuinely KB-helpful for LangPro on this Premise/Hypothesis pair.

{NEW_PROMPT_RULES}

## EXAMPLES

The examples below were chosen from cached current proof-useful checks or explicitly old-proof-grounded replay evidence.

### Example 1: Verb normalization
Premise: A man is strumming a guitar.
Hypothesis: A man is playing guitar.
[KB_START]
isa_wn(strum, play)
[KB_END]

### Example 2: Event paraphrase
Premise: A man vaults over a high bar.
Hypothesis: A man jumps.
[KB_START]
isa_wn(vault, jump)
[KB_END]

### Example 3: Old-proof-grounded broad activity bridge
Premise: A child rides a swing.
Hypothesis: A kid is playing.
[KB_START]
isa_wn(child, kid)
isa_wn(ride, play)
[KB_END]

### Example 4: disj only for genuine opposites
Premise: The window is open.
Hypothesis: The window is closed.
[KB_START]
disj(open, closed)
[KB_END]

Premise: ${{premise}}
Hypothesis: ${{hypothesis}}

Generate the knowledge injections:
"""

def _insert_before_marker(template: str, marker: str, addition: str) -> str:
    if marker not in template:
        raise ValueError(f"Marker not found in prompt template: {marker}")
    return template.replace(marker, f"{addition.rstrip()}\n\n{marker}", 1)


ETTORE_BASE_PROMPT = ICL_V2_TEMPLATE

ETTORE_PRECISION_CALIBRATION = """## CALIBRATION UPDATE: precision without changing the task

Keep all rules above. Apply these extra checks only to resolve ambiguity:
- Prefer no relation over a broad event or social inference.
- Do not infer activities from objects or situations, for example carrying a shopping bag is not enough for shop.
- Do not infer event paraphrases unless the two verbs are direct lexical paraphrases.
- Output at most one form of the same bridge; never output both an inflected phrase and its lemma.
- If a candidate relation only proves a sentence that is already syntactically entailed, omit it.
"""

ETTORE_PROMPT = _insert_before_marker(
    ETTORE_BASE_PROMPT,
    "Premise: ${premise}",
    ETTORE_PRECISION_CALIBRATION,
)

LASHA_PROMPT = '''You are an expert in linguistic semantics and logic. You will receive a Natural Language Inference (NLI) problem in English, consisting of a premise sentence and a hypothesis sentence.
You will reason carefully and decide whether the premise entails the hypothesis, which means that if the premise is true, then the hypothesis must also be true under ordinary English meaning and widely accepted background knowledge.
If the answer is "entailment", output a structured explanation that is a set of lexical entailment relations over short phrases that explain why the hypothesis is entailed from the premise.
Lexical entailment should be defined over short phrases that are lemmatized or normalized versions of short phrases occurring in the premise and the hypothesis, e.g., entails(phrase_1, phrase_2), and it means that phrase_1 is a type of phrase_2, for example, entails(woman, person), entails(dog, domestic animal), entails(huge, very big), and entails(run, move fast).
Use lexical entailment relations only when they are needed to explain the entailment. If the entailment follows without any non-trivial lexical relation, output an empty set.

Relation formatting rules:
The meaning of a lexical entailment relation has to be acceptable based on common sense, e.g., entails(woman, blond person) is not acceptable.
A lexical entailment may not express a trivial relation that is obtainable by discarding modifiers, e.g., entails(blond woman, woman) is not acceptable.
A lexical entailment may not contain redundant words such as auxiliary verbs and the infinitive "to", e.g., entails(will walk, will move), entails(is red, is colored), and entails(to walk, to move) are not acceptable.
Phrases in a lexical entailment have to contain lemmatized words, e.g., entails(dogs, domestic animals) is not acceptable.
Phrases in a lexical entailment may not contain determiners, e.g., entails(a dog, a domestic animal) is not acceptable.
Phrases in a lexical entailment may not contain prepositional phrases, e.g., entails(dog with spots, a domestic animal) is not acceptable.

Examples:
Example 1:
	input:
		premise: Young ladies are playing the guitar.
		hypothesis: A musical instrument is being played by girls.
	correct output:
		answer: entailment
		relations: { entails(young lady, girl), entails(guitar, musical instrument) }
	unwanted output:
		relations: { entails(young ladies, girls), entails(the guitar, a musical instrument) }
		explanation: phrases in relations may not have determiners, e.g., "a" and "the". "ladies" and "girls" have to use lemmas "lady" and "girl", respectively.

Example 2:
	input:
		premise: A female swimmer getting out of the pool still dripping wet.
		hypothesis: A woman gets out of the pool.
	correct output:
		answer: entailment
		relations: { entails(female swimmer, woman) }
	unwanted output:
		relations: { entails(swimmer, woman) }
		explanation: it is a factually wrong relation because not every swimmer is a woman

Example 3:
	input:
		premise: A young girl wearing a pink coat plays with a yellow toy.
		hypothesis: A kid is swinging a toy golf club.
	correct output:
		answer: non-entailment

Example 4:
	input:
		premise: A black race car starts up in front of a crowd of people.
		hypothesis: A car is running.
	correct output:
		answer: entailment
		relations: { entails(start up, run) }
	unwanted output:
		relations: { entails(starts up, is running) }
		explanation: "starts up" and "is running" do not contain lemmatized words, and "is" is unnecessary in "is running".

Example 5:
	input:
		premise: A woman dressed in red clothing is dancing inside a crowd of people.
		hypothesis: A woman in red is dancing in a crowd.
	correct output:
		answer: entailment
		relations: { }
	unwanted output:
		relations: { entails(in red clothing, in red) }
		explanation: The phrases in the relation may not include prepositional phrases, e.g., "in red clothing"

Example 6:
	input:
		premise: A tall man with a cap is climbing a cord.
		hypothesis: The man in a hat is climbing a rope.
	correct output:
		answer: entailment
		relations: { entails(cap, hat), entails(cord, rope) }
	unwanted output:
		relations: { entails(cap, hat), entails(tall man, man) }
		explanation: "entails(cord, rope)" is missing. entails(tall man, man) is trivial since it includes dropping the adjective "tall".

Example 7:
	input:
		premise: No person is cooking.
		hypothesis: No cook is cooking in the kitchen.
	correct output:
		answer: entailment
		relations: { entails(cook, person) }
	unwanted output:
		relations: { entails(person, cook) }
		explanation: The relation does not help to explain "entailment", taking into account that negation reverses a lexical entailment direction. It is also a factually wrong relation because not every person is a cook.

Example 8:
	input:
		premise: A person who is obese is holding a chinchilla.
		hypothesis: A fat person is holding a small animal.
	correct output:
		answer: entailment
		relations: { entails(chinchilla, small animal), entails(obese, fat) }
	unwanted output:
		relations: { entails(fat, obese), entails(chinchilla, animal) }
		explanation: "entails(fat, obese)" needs to reverse its arguments to align with the entailment direction. "entails(chinchilla, animal)" is not sufficient to explain "entailment" since it misses "small", which is crucial.

Example 9:
	input:
		premise: A little boy is laughing and happily bouncing on a trampoline outside.
		hypothesis: The child is jumping outdoors.
	correct output:
		answer: entailment
		relations: { entails(little boy, child), entails(bounce, jump), entails(outside, outdoors) }
	unwanted output:
		relations: { entails(boy, child), entails(outdoors, outside) }
		explanation: "entails(bounce, jump)" is missing. "entails(little boy, child)" is preferred over "entails(boy, child)" as the former is more acceptable. "entails(outdoors, outside)" needs to reverse its arguments to align it to the entailment direction.

Now process the following input while strictly following the above instructions and formatting.
Output exactly in one of the following formats:
	answer: entailment
	relations: { ... }
	or
	answer: non-entailment

input:
	premise: ${PREMISE}
	hypothesis: ${HYPOTHESIS}
correct output:'''

LASHA_BASE_PROMPT = LASHA_PROMPT

LASHA_PRECISION_CALIBRATION = """Additional calibration while preserving all rules above:
- Only output lexical entailment relations that are both factually acceptable and needed to explain the entailment.
- Do not output relations for entailments that follow without a non-trivial lexical bridge.
- Do not use event or social implications as lexical entailment unless the phrase relation is a direct paraphrase.
- Do not output modifier-dropping relations such as old woman -> woman or military men -> men.
- Do not output both an inflected form and a lemmatized form for the same relation.
- If the best relation would violate any existing formatting rule, omit it instead of approximating it.
"""

LASHA_PROMPT = _insert_before_marker(
    LASHA_BASE_PROMPT,
    "Now process the following input while strictly following the above instructions and formatting.",
    LASHA_PRECISION_CALIBRATION,
)

COT_V2_TEMPLATE = f"""You are a knowledge base injection assistant using chain-of-thought reasoning.

## TASK
Analyze the semantic gap between Premise and Hypothesis, then generate only the KB facts that are genuinely KB-helpful for LangPro on this pair.

{NEW_PROMPT_RULES}

## REASONING STEPS

1. Identify key terms in Premise.
2. Identify key terms in Hypothesis.
3. Find only the semantic gaps that actually need a bridge.
4. Decide whether each gap calls for isa_wn, disj, or no fact at all.
5. Keep the final set minimal and concretely helpful for LangPro.
6. Reason step-by-step internally, then output only the KB facts in delimiters.

Premise: ${{premise}}
Hypothesis: ${{hypothesis}}

Generate the knowledge injections:
"""

# Initial KB-generation prompt from the agentic-pipeline-langpro repository.
# The placeholders use this module's ${premise}/${hypothesis} convention so it
# can be evaluated by the standard multi-reference experiment runner.
STEFAN_PROMPT = """You are an expert in linguistic semantics and logic. You will receive a Natural Language Inference (NLI) problem in English, consisting of a premise sentence and a hypothesis sentence.
You will reason carefully and decide whether the premise entails the hypothesis, which means that if the premise is true, then the hypothesis must also be true under ordinary English meaning and widely accepted background knowledge.
If the answer is "entailment", output a structured explanation that is a set of lexical entailment relations over short phrases that explain why the hypothesis is entailed from the premise. If the answer is not "entailment", output an empty KB.
Lexical entailment should be defined over short phrases that are lemmatized or normalized versions of short phrases occurring in the premise and the hypothesis, e.g., isa_wn(phrase_1, phrase_2), and it means that phrase_1 is a type of phrase_2, for example, isa_wn(woman, person), isa_wn(dog, domestic animal), isa_wn(huge, very big), and isa_wn(run, move fast).
Use lexical entailment relations only when they are needed to explain the entailment. If the entailment follows without any non-trivial lexical relation, output an empty set.

Relation formatting rules:
The meaning of a lexical entailment relation has to be acceptable based on common sense, e.g., isa_wn(woman, blond person) is not acceptable.
A lexical entailment may not express a trivial relation that is obtainable by discarding modifiers, e.g., isa_wn(blond woman, woman) is not acceptable.
A lexical entailment may not contain redundant words such as auxiliary verbs and the infinitive "to", e.g., isa_wn(will walk, will move), isa_wn(is red, is colored), and isa_wn(to walk, to move) are not acceptable.
Phrases in a lexical entailment have to contain lemmatized words, e.g., isa_wn(dogs, domestic animals) is not acceptable.
Phrases in a lexical entailment may not contain determiners, e.g., isa_wn(a dog, a domestic animal) is not acceptable.
Phrases in a lexical entailment may not contain prepositional phrases, e.g., isa_wn(dog with spots, a domestic animal) are not acceptable.

Examples:
Example 1:
    input:
        premise: Young ladies are playing the guitar.
        hypothesis: A musical instrument is being played by girls.
    correct output:
        [KB_START]
        isa_wn(young lady, girl)
        isa_wn(guitar, musical instrument)
        [KB_END]
    unwanted output:
        [KB_START]
        isa_wn(young ladies, girls)
        isa_wn(the guitar, a musical instrument)
        [KB_END]
        explanation: phrases in relations may not have determiners, e.g., "a" and "the". "ladies" and "girls" have to use lemmas "lady" and "girl", respectively.

Example 2:
    input:
        premise: A female swimmer getting out of the pool still dripping wet.
        hypothesis: A woman gets out of the pool.
    correct output:
        [KB_START]
        isa_wn(female swimmer, woman)
        [KB_END]
    unwanted output:
        [KB_START]
        isa_wn(swimmer, woman)
        [KB_END]
        explanation: it is a factually wrong relation because not every swimmer is a woman

Example 3:
    input:
        premise: A young girl wearing a pink coat plays with a yellow toy.
        hypothesis: A kid is swinging a toy golf club.
    correct output:
        [KB_START]
        [KB_END]

Example 4:
    input:
        premise: A black race car starts up in front of a crowd of people.
        hypothesis: A car is running.
    correct output:
        [KB_START]
        isa_wn(start up, run)
        [KB_END]
    unwanted output:
        [KB_START]
        isa_wn(starts up, is running)
        [KB_END]
        explanation: "starts up" and "is running" do not contain lemmatized words, and "is" is unnecessary in "is running".

Example 5:
    input:
        premise: A woman dressed in red clothing is dancing inside a crowd of people.
        hypothesis: A woman in red is dancing in a crowd.
    correct output:
        [KB_START]
        [KB_END]
    unwanted output:
        [KB_START]
        isa_wn(in red clothing, in red)
        [KB_END]
        explanation: The phrases in the relation may not include prepositional phrases, e.g., "in red clothing"

Example 6:
    input:
        premise: A tall man with a cap is climbing a cord.
        hypothesis: The man in a hat is climbing a rope.
    correct output:
        [KB_START]
        isa_wn(cap, hat)
        isa_wn(cord, rope)
        [KB_END]
    unwanted output:
        [KB_START]
        isa_wn(cap, hat)
        isa_wn(tall man, man)
        [KB_END]
        explanation: "isa_wn(cord, rope)" is missing. isa_wn(tall man, man) is trivial since it includes dropping the adjective "tall".

Example 7:
    input:
        premise: No person is cooking.
        hypothesis: No cook is cooking in the kitchen.
    correct output:
        [KB_START]
        isa_wn(cook, person)
        [KB_END]
    unwanted output:
        [KB_START]
        isa_wn(person, cook)
        [KB_END]
        explanation: The relation does not help to explain "entailment", taking into account that negation reverses a lexical entailment direction. It is also a factually wrong relation because not every person is a cook.

Example 8:
    input:
        premise: A person who is obese is holding a chinchilla.
        hypothesis: A fat person is holding a small animal.
    correct output:
        [KB_START]
        isa_wn(chinchilla, small animal)
        isa_wn(obese, fat)
        [KB_END]
    unwanted output:
        [KB_START]
        isa_wn(fat, obese)
        isa_wn(chinchilla, animal)
        [KB_END]
        explanation: "isa_wn(fat, obese)" needs to reverse its arguments to align with the entailment direction. "isa_wn(chinchilla, animal)" is not sufficient to explain "entailment" since it misses "small", which is crucial.

Example 9:
    input:
        premise: A little boy is laughing and happily bouncing on a trampoline outside.
        hypothesis: The child is jumping outdoors.
    correct output:
        [KB_START]
        isa_wn(little boy, child)
        isa_wn(bounce, jump)
        isa_wn(outside, outdoors)
        [KB_END]
    unwanted output:
        [KB_START]
        isa_wn(boy, child)
        isa_wn(outdoors, outside)
        [KB_END]
        explanation: "isa_wn(bounce, jump)" is missing. "isa_wn(little boy, child)" is preferred over "isa_wn(boy, child)" as the former is more acceptable. "isa_wn(outdoors, outside)" needs to reverse its arguments to align it to the entailment direction.

Additional calibration while preserving all rules above:
- Only output lexical entailment relations that are both factually acceptable and needed to explain the entailment.
- Do not output relations for entailments that follow without a non-trivial lexical bridge.
- Do not use event or social implications as lexical entailment unless the phrase relation is a direct paraphrase.
- Do not output modifier-dropping relations such as old woman -> woman or military men -> men.
- Do not output both an inflected form and a lemmatized form for the same relation.
- If the best relation would violate any existing formatting rule, omit it instead of approximating it.

Now process the following input while strictly following the above instructions and formatting.
Output exactly in the following format:
    [KB_START]
    predicate(arg1, arg2)
    [KB_END]

input:
    premise: ${premise}
    hypothesis: ${hypothesis}
"""

# Combined prompt registry (for backward compatibility, maps old names to legacy)
prompts = {
    "prompts": [
        # Legacy prompts
        {
            "name": "legacy_cot",
            "description": "[LEGACY] Chain-of-thought prompt (use cot for improved version)",
            "template": legacy_prompts["prompts"][0]["template"],
        },
        {
            "name": "legacy_least_to_most",
            "description": "[LEGACY] Least-to-most decomposition prompt",
            "template": legacy_prompts["prompts"][1]["template"],
        },
        {
            "name": "legacy_icl",
            "description": "[LEGACY] In-context learning prompt (use icl for improved version)",
            "template": legacy_prompts["prompts"][2]["template"],
        },
        # Improved prompts (now default)
        {
            "name": "icl",
            "description": "Improved ICL with shared rules and in-context examples",
            "template": ICL_V2_TEMPLATE,
        },
        {
            "name": "ettore",
            "description": "Alias for Ettore's improved ICL prompt",
            "template": ETTORE_PROMPT,
        },
        {
            "name": "lasha",
            "description": "Lasha lexical-entailment ICL prompt",
            "template": LASHA_PROMPT,
        },
        {
            "name": "stefan",
            "description": "Stefan's agentic initial lexical-entailment prompt",
            "template": STEFAN_PROMPT,
        },
        {
            "name": "cot",
            "description": "Improved CoT with shared rules and structured reasoning",
            "template": COT_V2_TEMPLATE,
        },
    ]
}


def get_prompt(prompt_name: str) -> str:
    """Get a prompt template by name from the in-memory dictionary `prompts`."""
    for prompt in prompts.get("prompts", []):
        if prompt.get("name") == prompt_name:
            return prompt.get("template", "")
    raise KeyError(
        f"Prompt '{prompt_name}' not found. Available: {[p['name'] for p in prompts['prompts']]}"
    )


def fill_prompt(prompt_name: str, premises: list, hypothesis: str) -> str:
    """Get a prompt template and fill in the premises and hypothesis."""
    template_str = get_prompt(prompt_name)
    premise_text = "\n".join(premises) if isinstance(premises, list) else premises
    return (
        template_str
        .replace("${premise}", premise_text)
        .replace("${hypothesis}", hypothesis)
        .replace("${PREMISE}", premise_text)
        .replace("${HYPOTHESIS}", hypothesis)
    )


def list_prompts() -> list:
    """Return a list of available prompt names."""
    return [p["name"] for p in prompts.get("prompts", [])]
