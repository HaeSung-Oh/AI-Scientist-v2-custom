import argparse
import json
import os.path as osp
import re
import traceback
from typing import Any, Dict, List

import sys

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from ai_scientist.llm import (
    AVAILABLE_LLMS,
    create_client,
    get_response_from_llm,
)

from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool
from ai_scientist.tools.literature_search import (
    ArxivSearchTool,
    OpenAlexSearchTool,
    PubMedSearchTool,
    SerpApiSearchTool,
)
from ai_scientist.tools.base_tool import BaseTool

# Create tool instances
semantic_scholar_tool = SemanticScholarSearchTool()
arxiv_tool = ArxivSearchTool()
pubmed_tool = PubMedSearchTool()
openalex_tool = OpenAlexSearchTool()
google_scholar_tool = SerpApiSearchTool(scholarly=True)
web_search_tool = SerpApiSearchTool(scholarly=False)

# Define tools at the top of the file
tools = [
    semantic_scholar_tool,
    arxiv_tool,
    pubmed_tool,
    openalex_tool,
    google_scholar_tool,
    web_search_tool,
    {
        "name": "FinalizeIdea",
        "description": """Finalize your idea by providing the idea details.

The IDEA JSON should include the following fields:
- "Name": A short descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A catchy and informative title for the proposal.
- "Short Hypothesis": A concise statement of the main hypothesis or research question. Clarify the need for this specific direction, ensure this is the best setting to investigate this idea, and there are not obvious other simpler ways to answer the question.
- "Novelty Check": A concrete explanation of why this is not just a stale re-use of familiar baselines, a shallow module swap, or a minor variant of existing work. Cite the recent search evidence that shaped the claim.
- "Evidence Used": A list of at least three concrete pieces of search evidence used to shape the proposal. Each item should include the source/tool, title or result name, year if available, and why it matters.
- "FINER Scores": Scores from 1-10 for Feasible, Interesting, Novel, Ethical, and Relevant, with one concise justification per score.
- "Scope": A concrete scope definition with In Scope and Out of Scope lists.
- "Candidate Alternatives Considered": A list of at least three alternative ideas considered and why each was rejected.
- "Related Work": A brief discussion of the most relevant related work and how the proposal clearly distinguishes from it, and is not a trivial extension.
- "Abstract": An abstract that summarizes the proposal in conference format (approximately 250 words).
- "Experiments": A list of experiments that would be conducted to validate the proposal. Ensure these are simple and feasible. Be specific in exactly how you would test the hypothesis, and detail precise algorithmic changes. Include the evaluation metrics you would use.
- "Validation Plan": A concrete plan for testing whether the idea is actually useful, including baselines, ablations, failure cases, and what result would falsify the hypothesis.
- "Risk Factors and Limitations": A list of potential risks and limitations of the proposal.""",
    },
]

# Create a tools dictionary for easy lookup
tools_dict = {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}

# Create a string with the tool descriptions
tool_descriptions = "\n\n".join(
    (
        f"- **{tool.name}**: {tool.description}"
        if isinstance(tool, BaseTool)
        else f"- **{tool['name']}**: {tool['description']}"
    )
    for tool in tools
)

# Extract tool names for the prompt
tool_names = [
    f'"{tool.name}"' if isinstance(tool, BaseTool) else f'"{tool["name"]}"'
    for tool in tools
]
tool_names_str = ", ".join(tool_names)

system_prompt = f"""You are an experienced AI researcher who aims to propose high-impact research ideas resembling exciting grant proposals. Feel free to propose any novel ideas or experiments; make sure they are novel. Be very creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis about the topic. For example, they could involve very interesting and simple interventions or investigations that explore new possibilities or challenge existing assumptions. Clearly clarify how the proposal distinguishes from the existing literature.

Ensure that the proposal does not require resources beyond what an academic lab could afford. These proposals should lead to papers that are publishable at top ML conferences.

Avoid stale, generic ideas that merely repackage familiar baselines, add a shallow module, or make an incremental architecture swap without a clear scientific hypothesis. Use literature searches to discover current research directions instead of relying on predefined method keywords. Prefer ideas whose motivation, mechanism, and validation plan are grounded in recent evidence and a careful reading of the target problem.

Finding overlap with a broad method family does not automatically invalidate an idea. Treat overlap as a boundary-setting signal: identify what is already covered, then refine the proposal around a distinct mechanism, setting, evaluation protocol, dataset regime, or hypothesis.

Synthetic, procedurally generated, or simulated data may be used only for debugging, smoke tests, pipeline checks, or controlled sanity checks. It must not be used as the main validation evidence for the research claim. The final validation plan must use real public datasets whenever suitable datasets are available, and must report metrics on real data. Use any domain-specific dataset requirements stated in the workshop description. Any proposal that validates the core claim only on synthetic, procedurally generated, or simulated data is invalid and must be revised before FinalizeIdea.

Use deep internal thinking before choosing each action. If the model supports Qwen thinking mode, use it. Compare multiple candidate directions, reject weak or stale variants, and stress-test the chosen idea before finalizing. Your final visible response must still contain only ACTION and ARGUMENTS in the required format. Do not print thinking, chain-of-thought, markdown fences, headings, follow-up reasoning, or extra commentary.

You have access to the following tools:

{tool_descriptions}

Respond in the following format:

ACTION:
<The action to take, exactly one of {tool_names_str}>

ARGUMENTS:
<If ACTION is a search tool, provide the search query as {{"query": "[PURPOSE] your search query"}} where PURPOSE is one of [BROAD], [OVERLAP], [DISCONFIRM], or [DATASET]. If ACTION is "FinalizeIdea", provide the idea details as {{"idea": {{ ... }}}} with the IDEA JSON specified below.>

If you choose to finalize your idea, provide the IDEA JSON in the arguments:

IDEA JSON:
```json
{{
  "idea": {{
    "Name": "...",
    "Title": "...",
    "Short Hypothesis": "...",
    "Novelty Check": "...",
    "Evidence Used": [
      {{"source": "...", "title": "...", "year": "...", "why_it_matters": "..."}},
      {{"source": "...", "title": "...", "year": "...", "why_it_matters": "..."}},
      {{"source": "...", "title": "...", "year": "...", "why_it_matters": "..."}}
    ],
    "FINER Scores": {{
      "Feasible": {{"score": 1, "justification": "..."}},
      "Interesting": {{"score": 1, "justification": "..."}},
      "Novel": {{"score": 1, "justification": "..."}},
      "Ethical": {{"score": 1, "justification": "..."}},
      "Relevant": {{"score": 1, "justification": "..."}}
    }},
    "Scope": {{"In Scope": ["..."], "Out of Scope": ["..."]}},
    "Candidate Alternatives Considered": ["...", "...", "..."],
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Validation Plan": "...",
    "Risk Factors and Limitations": "..."
  }}
}}
```

Ensure the JSON is properly formatted for automatic parsing.

Note: You must satisfy the runtime literature-search requirements before finalizing your idea. Prefer mixing search sources, for example Semantic Scholar plus PubMed for biomedical grounding, or arXiv plus OpenAlex for recent ML methods."""

# Define the initial idea generation prompt
idea_generation_prompt = """{workshop_description}

/think

Here are the proposals that you have already generated:

'''
{prev_ideas_string}
'''

Begin by deeply investigating an interestingly new high-level research proposal that differs from what you have previously proposed. Do not finalize immediately unless the literature search and validation reasoning are already strong enough. Prefer searching first.

Across the search process, cover diverse search purposes. Prefix every search query with exactly one purpose tag:
- [BROAD] for broad landscape discovery
- [OVERLAP] for exact task or contribution overlap
- [DISCONFIRM] for novelty-disconfirming searches that could weaken the idea
- [DATASET] for dataset, benchmark, metric, and evaluation feasibility

Shared transcript so far:
{shared_transcript}

Evidence bank so far:
{evidence_bank}

Search purpose status:
{search_purpose_status}

{search_requirements}
"""

# Define the reflection prompt
idea_reflection_prompt = """Round {current_round}/{num_reflections}.

/think

Workshop description and constraints:
{workshop_description}

In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
Include any other factors that you think are important in evaluating the proposal.
Ensure the proposal is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your proposal.
Stick to the spirit of the original idea unless there are glaring issues.

Before choosing your next action, explicitly check whether the idea would still look non-obvious and current relative to 2024-2026 work. If the idea mainly relies on a familiar baseline plus a small tweak, revise it or search for broader alternatives instead of finalizing.
Before finalizing, compare at least three candidate ideas, reject the weaker ones, and define a validation plan with baselines, ablations, expected evidence, and falsification criteria.

If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.

Results from your last action (if any):

{last_tool_results}

Across the search process, cover diverse search purposes. Prefix every search query with exactly one purpose tag:
- [BROAD] for broad landscape discovery
- [OVERLAP] for exact task or contribution overlap
- [DISCONFIRM] for novelty-disconfirming searches that could weaken the idea
- [DATASET] for dataset, benchmark, metric, and evaluation feasibility

Shared transcript so far:
{shared_transcript}

Evidence bank so far:
{evidence_bank}

Search purpose status:
{search_purpose_status}

Current literature-search requirement/status:
{search_requirements}
"""

agent_role_prompts = {
    "IdeationAgent": """You are the IdeationAgent.
Work as a single tool-using research ideation agent. Search, reflect, and finalize only when the evidence and validation plan are strong enough.""",
    "ExplorerAgent": """You are the ExplorerAgent.
Broaden the search space, identify surprising research directions, and choose literature searches that reveal what is current and underexplored. Prefer searching when evidence is thin. Do not finalize unless the shared transcript already contains enough search evidence and candidate comparison.""",
    "SkepticAgent": """You are the SkepticAgent.
Challenge novelty, feasibility, evaluation realism, and hidden overlap with prior work. Prefer searches that could disconfirm the current idea or reveal stronger related work. If the proposal looks incremental, choose a search or force a sharper distinction instead of finalizing.""",
    "ExperimentAgent": """You are the ExperimentAgent.
Focus on concrete datasets, metrics, baselines, ablations, resource limits, and falsification criteria. Prefer searches that clarify whether the validation plan is realistic on public data. Do not finalize vague ideas that cannot be tested cleanly.""",
    "SynthesizerAgent": """You are the SynthesizerAgent.
Integrate the search evidence and critiques into one clean proposal. Finalize only when the search requirements are satisfied, at least three candidate directions have been compared, the Evidence Used field cites concrete items from the evidence bank, and the validation plan is concrete. Otherwise choose the most useful next search.""",
}


def get_agent_schedule(agent_mode: str) -> List[str]:
    if agent_mode == "single":
        return ["IdeationAgent"]
    return ["ExplorerAgent", "SkepticAgent", "ExperimentAgent", "SynthesizerAgent"]


def build_agent_system_prompt(agent_name: str) -> str:
    return f"{system_prompt}\n\n{agent_role_prompts[agent_name]}"


def trim_for_transcript(text: Any, max_chars: int = 1800) -> str:
    compact = " ".join(str(text).split())
    return compact[:max_chars] + ("..." if len(compact) > max_chars else "")


def format_shared_transcript(entries: List[str], max_chars: int = 7000) -> str:
    if not entries:
        return "No shared transcript yet."

    selected = []
    total_chars = 0
    for entry in reversed(entries):
        entry_len = len(entry)
        if selected and total_chars + entry_len > max_chars:
            break
        selected.append(entry)
        total_chars += entry_len
    return "\n".join(reversed(selected))


def extract_evidence_items(action: str, query: str, result: str, max_items: int = 4) -> List[Dict]:
    evidence_items = []
    if not search_result_is_successful(result):
        return evidence_items

    for line in result.splitlines():
        match = re.match(r"^\s*\d+:\s*(.+?)(?:\.\s|$)", line.strip())
        if not match:
            continue
        title = match.group(1).strip()
        if not title:
            continue
        year_match = re.search(r"\b(20\d{2})\b", line)
        evidence_items.append(
            {
                "source": action,
                "query": query,
                "title": title[:220],
                "year": year_match.group(1) if year_match else "Unknown",
            }
        )
        if len(evidence_items) >= max_items:
            break
    return evidence_items


def format_evidence_bank(evidence_bank: List[Dict], max_items: int = 18) -> str:
    if not evidence_bank:
        return "No evidence collected yet."

    lines = []
    for item in evidence_bank[-max_items:]:
        lines.append(
            "- "
            f"{item.get('source', 'Unknown source')} | "
            f"{item.get('year', 'Unknown year')} | "
            f"{item.get('title', 'Unknown title')} | "
            f"query: {item.get('query', '')}"
        )
    return "\n".join(lines)


SEARCH_PURPOSES = ["BROAD", "OVERLAP", "DISCONFIRM", "DATASET"]


def parse_search_purpose(query: str) -> tuple[str | None, str]:
    match = re.match(r"^\s*\[(BROAD|OVERLAP|DISCONFIRM|DATASET)\]\s*(.*)$", query)
    if not match:
        return None, query.strip()
    return match.group(1), match.group(2).strip()


def format_search_purpose_status(successful_search_purposes: Dict[str, int]) -> str:
    return "; ".join(
        f"{purpose}: {successful_search_purposes.get(purpose, 0)}"
        for purpose in SEARCH_PURPOSES
    )


def normalize_for_match(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def evidence_used_matches_bank(evidence_used: List[Any], evidence_bank: List[Dict]) -> int:
    bank_titles = [normalize_for_match(item.get("title", "")) for item in evidence_bank]
    matches = 0
    for evidence in evidence_used:
        if isinstance(evidence, dict):
            evidence_text = " ".join(
                str(evidence.get(key, ""))
                for key in ["title", "source", "year", "why_it_matters"]
            )
        else:
            evidence_text = str(evidence)
        normalized_evidence = normalize_for_match(evidence_text)
        if not normalized_evidence:
            continue
        if any(
            title and (title in normalized_evidence or normalized_evidence in title)
            for title in bank_titles
        ):
            matches += 1
    return matches


def parse_json_prefix(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```json"):
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            text = text.removeprefix("```").strip()

    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    return json.JSONDecoder().raw_decode(text[start:])[0]


def parse_action_response(response_text: str) -> tuple[str, str]:
    action_pattern = r"ACTION:\s*(.*?)\s*ARGUMENTS:"
    arguments_pattern = r"ARGUMENTS:\s*(.*?)(?:$|\nTHOUGHT:|\nACTION:|\n$)"

    action_match = re.search(action_pattern, response_text, re.DOTALL | re.IGNORECASE)
    arguments_match = re.search(
        arguments_pattern, response_text, re.DOTALL | re.IGNORECASE
    )

    if all([action_match, arguments_match]):
        return action_match.group(1).strip(), arguments_match.group(1).strip()

    try:
        arguments_json = parse_json_prefix(response_text)
    except json.JSONDecodeError:
        pass
    else:
        if "idea" in arguments_json:
            return "FinalizeIdea", json.dumps(arguments_json)
        if "query" in arguments_json:
            return "SearchSemanticScholar", json.dumps(arguments_json)

    raise ValueError("Failed to parse the LLM response.")


def validate_final_idea(idea: Dict) -> List[str]:
    required_fields = [
        "Name",
        "Title",
        "Short Hypothesis",
        "Novelty Check",
        "Evidence Used",
        "FINER Scores",
        "Scope",
        "Candidate Alternatives Considered",
        "Related Work",
        "Abstract",
        "Experiments",
        "Validation Plan",
        "Risk Factors and Limitations",
    ]
    missing = [field for field in required_fields if not idea.get(field)]

    alternatives = idea.get("Candidate Alternatives Considered")
    if not isinstance(alternatives, list) or len(alternatives) < 3:
        missing.append("at least three Candidate Alternatives Considered")

    evidence_used = idea.get("Evidence Used")
    if not isinstance(evidence_used, list) or len(evidence_used) < 3:
        missing.append("at least three Evidence Used items")

    finer_scores = idea.get("FINER Scores")
    if not isinstance(finer_scores, dict):
        missing.append("FINER Scores object")
    else:
        for key in ["Feasible", "Interesting", "Novel", "Ethical", "Relevant"]:
            score_item = finer_scores.get(key)
            if isinstance(score_item, dict):
                score = score_item.get("score")
                justification = score_item.get("justification")
            else:
                score = score_item
                justification = None
            if not isinstance(score, (int, float)) or not 1 <= score <= 10:
                missing.append(f"FINER Scores.{key} score from 1-10")
            if not justification:
                missing.append(f"FINER Scores.{key} justification")

    scope = idea.get("Scope")
    if not isinstance(scope, dict):
        missing.append("Scope object")
    else:
        in_scope = scope.get("In Scope") or scope.get("in_scope")
        out_scope = scope.get("Out of Scope") or scope.get("out_of_scope")
        if not isinstance(in_scope, list) or not in_scope:
            missing.append("Scope In Scope list")
        if not isinstance(out_scope, list) or not out_scope:
            missing.append("Scope Out of Scope list")

    experiments = idea.get("Experiments")
    if not isinstance(experiments, list) or len(experiments) < 3:
        missing.append("at least three Experiments")

    validation_plan = str(idea.get("Validation Plan", "")).lower()
    for term in ["baseline", "ablation"]:
        if term not in validation_plan:
            missing.append(f"Validation Plan mentioning {term}")

    novelty_check = str(idea.get("Novelty Check", "")).lower()
    if "2024" not in novelty_check and "2025" not in novelty_check and "2026" not in novelty_check:
        missing.append("Novelty Check grounded in 2024-2026 evidence")

    return missing


def search_result_is_successful(result: str) -> bool:
    if not result:
        return False

    result_lower = result.strip().lower()
    failure_markers = [
        "unavailable:",
        "error using tool",
        "no papers found",
        "no arxiv papers found",
        "no pubmed papers found",
        "no openalex works found",
        "no searchgooglescholar results found",
        "no searchweb results found",
    ]
    return not any(marker in result_lower for marker in failure_markers)


def generate_temp_free_idea(
    idea_fname: str,
    client: Any,
    model: str,
    workshop_description: str,
    max_num_generations: int = 20,
    num_reflections: int = 5,
    min_num_searches: int = 4,
    min_num_search_sources: int = 3,
    min_num_search_purposes: int = 3,
    max_tokens: int | None = None,
    agent_mode: str = "single",
    reload_ideas: bool = True,
) -> List[Dict]:
    idea_str_archive = []
    # load ideas from file
    if reload_ideas and osp.exists(idea_fname):
        with open(idea_fname, "r") as f:
            idea_str_content = json.load(f)
            for idea in idea_str_content:
                idea_str_archive.append(json.dumps(idea))
            print(f"Loaded {len(idea_str_archive)} ideas from {idea_fname}")
    else:
        print(f"No ideas found in {idea_fname}. Starting from scratch.")

    for gen_idx in range(max_num_generations):
        print()
        print(f"Generating proposal {gen_idx + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            last_tool_results = ""
            idea_finalized = False
            agent_schedule = get_agent_schedule(agent_mode)
            agent_histories = {agent_name: [] for agent_name in agent_schedule}
            shared_transcript = []
            evidence_bank = []
            successful_searches = 0
            successful_search_sources = set()
            successful_search_purposes = {purpose: 0 for purpose in SEARCH_PURPOSES}
            total_agent_rounds = num_reflections * len(agent_schedule)

            for reflection_round in range(total_agent_rounds):
                agent_name = agent_schedule[reflection_round % len(agent_schedule)]
                cycle_idx = reflection_round // len(agent_schedule) + 1
                agent_step_idx = reflection_round % len(agent_schedule) + 1
                shared_transcript_text = format_shared_transcript(shared_transcript)
                evidence_bank_text = format_evidence_bank(evidence_bank)
                search_purpose_status = format_search_purpose_status(
                    successful_search_purposes
                )
                search_requirements = (
                    f"Before FinalizeIdea, perform at least {min_num_searches} "
                    f"successful literature searches from at least "
                    f"{min_num_search_sources} distinct search sources. Searches "
                    "that return no results, API-key unavailable messages, or errors "
                    "do not count. Current progress: "
                    f"{successful_searches}/{min_num_searches} successful searches; "
                    f"{len(successful_search_sources)}/{min_num_search_sources} "
                    f"sources used ({', '.join(sorted(successful_search_sources)) or 'none'}); "
                    f"{sum(1 for count in successful_search_purposes.values() if count > 0)}/"
                    f"{min_num_search_purposes} search purposes covered."
                )
                if reflection_round == 0:
                    # Use the initial idea generation prompt
                    prompt_text = idea_generation_prompt.format(
                        workshop_description=workshop_description,
                        prev_ideas_string=prev_ideas_string,
                        shared_transcript=shared_transcript_text,
                        evidence_bank=evidence_bank_text,
                        search_purpose_status=search_purpose_status,
                        search_requirements=search_requirements,
                    )
                else:
                    # Use the reflection prompt, including tool results if any
                    prompt_text = idea_reflection_prompt.format(
                        current_round=cycle_idx,
                        num_reflections=num_reflections,
                        workshop_description=workshop_description,
                        last_tool_results=last_tool_results or "No new results.",
                        shared_transcript=shared_transcript_text,
                        evidence_bank=evidence_bank_text,
                        search_purpose_status=search_purpose_status,
                        search_requirements=search_requirements,
                    )

                if agent_mode == "multi":
                    print(
                        f"Agent: {agent_name} "
                        f"(cycle {cycle_idx}/{num_reflections}, "
                        f"step {agent_step_idx}/{len(agent_schedule)})"
                    )
                else:
                    print(f"Agent: {agent_name}")
                response_text, agent_histories[agent_name] = get_response_from_llm(
                    prompt=prompt_text,
                    client=client,
                    model=model,
                    system_message=build_agent_system_prompt(agent_name),
                    msg_history=agent_histories[agent_name],
                    max_tokens=max_tokens,
                )

                # Parse the LLM's response
                try:
                    action, arguments_text = parse_action_response(response_text)
                    print(f"Action: {action}")
                    print(f"Arguments: {arguments_text}")

                    # Process the action and arguments
                    if action in tools_dict:
                        # It's a tool we have defined
                        tool = tools_dict[action]
                        # Parse arguments
                        try:
                            arguments_json = parse_json_prefix(arguments_text)
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid arguments JSON for {action}.")
                        query = str(arguments_json.get("query", ""))
                        search_purpose, clean_query = parse_search_purpose(query)
                        if action.startswith("Search") and search_purpose is None:
                            last_tool_results = (
                                "Search rejected: prefix the query with exactly one "
                                "purpose tag: [BROAD], [OVERLAP], [DISCONFIRM], or "
                                "[DATASET]."
                            )
                            print(last_tool_results)
                            shared_transcript.append(
                                f"[{agent_name} round {reflection_round + 1}] "
                                f"Action={action}; Rejected={trim_for_transcript(last_tool_results)}"
                            )
                            continue
                        if action.startswith("Search"):
                            arguments_json["query"] = clean_query

                        # Use the tool
                        try:
                            # Assuming the arguments match the parameters of the tool
                            result = tool.use_tool(**arguments_json)
                            last_tool_results = result
                            shared_transcript.append(
                                f"[{agent_name} round {reflection_round + 1}] "
                                f"Action={action}; Arguments={trim_for_transcript(arguments_text, 500)}; "
                                f"Result={trim_for_transcript(result)}"
                            )
                            if action.startswith("Search"):
                                if search_result_is_successful(result):
                                    successful_searches += 1
                                    successful_search_sources.add(action)
                                    successful_search_purposes[search_purpose] += 1
                                    evidence_bank.extend(
                                        extract_evidence_items(
                                            action=action,
                                            query=query,
                                            result=result,
                                        )
                                    )
                                    print(
                                        "Successful literature searches: "
                                        f"{successful_searches}/{min_num_searches}; "
                                        "sources: "
                                        f"{len(successful_search_sources)}/"
                                        f"{min_num_search_sources}"
                                    )
                                else:
                                    print(
                                        "Search did not count toward the requirement "
                                        "because it returned no usable results."
                                    )
                        except Exception as e:
                            last_tool_results = f"Error using tool {action}: {str(e)}"
                            shared_transcript.append(
                                f"[{agent_name} round {reflection_round + 1}] "
                                f"Action={action}; Error={trim_for_transcript(last_tool_results)}"
                            )
                    elif action == "FinalizeIdea":
                        if agent_mode == "multi" and agent_name != "SynthesizerAgent":
                            last_tool_results = (
                                "FinalizeIdea rejected: in multi-agent mode, only "
                                "SynthesizerAgent may finalize after integrating the "
                                "other agents' evidence and critiques."
                            )
                            print(last_tool_results)
                            shared_transcript.append(
                                f"[{agent_name} round {reflection_round + 1}] "
                                f"FinalizeIdea rejected; Reason={trim_for_transcript(last_tool_results)}"
                            )
                            continue
                        if (
                            successful_searches < min_num_searches
                            or len(successful_search_sources) < min_num_search_sources
                        ):
                            last_tool_results = (
                                "FinalizeIdea rejected: perform at least "
                                f"{min_num_searches} successful literature searches "
                                f"from at least {min_num_search_sources} distinct "
                                "search sources before finalizing. Searches that "
                                "return no results, API-key unavailable messages, or "
                                "errors do not count. Current progress: "
                                f"{successful_searches}/{min_num_searches} successful "
                                "searches; "
                                f"{len(successful_search_sources)}/"
                                f"{min_num_search_sources} sources."
                            )
                            print(last_tool_results)
                            shared_transcript.append(
                                f"[{agent_name} round {reflection_round + 1}] "
                                f"FinalizeIdea rejected; Reason={trim_for_transcript(last_tool_results)}"
                            )
                            continue
                        covered_purposes = sum(
                            1 for count in successful_search_purposes.values() if count > 0
                        )
                        if covered_purposes < min_num_search_purposes:
                            last_tool_results = (
                                "FinalizeIdea rejected: cover at least "
                                f"{min_num_search_purposes} distinct search purposes "
                                "before finalizing. Current purpose status: "
                                f"{format_search_purpose_status(successful_search_purposes)}."
                            )
                            print(last_tool_results)
                            shared_transcript.append(
                                f"[{agent_name} round {reflection_round + 1}] "
                                f"FinalizeIdea rejected; Reason={trim_for_transcript(last_tool_results)}"
                            )
                            continue
                        if len(evidence_bank) < 3:
                            last_tool_results = (
                                "FinalizeIdea rejected: fewer than three concrete "
                                "evidence items were extracted from successful searches."
                            )
                            print(last_tool_results)
                            shared_transcript.append(
                                f"[{agent_name} round {reflection_round + 1}] "
                                f"FinalizeIdea rejected; Reason={trim_for_transcript(last_tool_results)}"
                            )
                            continue

                        # Parse arguments
                        try:
                            arguments_json = parse_json_prefix(arguments_text)
                            idea = arguments_json.get("idea")
                            if not idea:
                                raise ValueError("Missing 'idea' in arguments.")
                            missing_requirements = validate_final_idea(idea)
                            if missing_requirements:
                                last_tool_results = (
                                    "FinalizeIdea rejected: missing or weak requirements: "
                                    + "; ".join(missing_requirements)
                                )
                                print(last_tool_results)
                                shared_transcript.append(
                                    f"[{agent_name} round {reflection_round + 1}] "
                                    f"FinalizeIdea rejected; Reason={trim_for_transcript(last_tool_results)}"
                                )
                                continue
                            evidence_matches = evidence_used_matches_bank(
                                idea.get("Evidence Used", []), evidence_bank
                            )
                            if evidence_matches < 2:
                                last_tool_results = (
                                    "FinalizeIdea rejected: at least two Evidence Used "
                                    "items must match concrete titles in the evidence bank."
                                )
                                print(last_tool_results)
                                shared_transcript.append(
                                    f"[{agent_name} round {reflection_round + 1}] "
                                    f"FinalizeIdea rejected; Reason={trim_for_transcript(last_tool_results)}"
                                )
                                continue

                            # Append the idea to the archive
                            idea_str_archive.append(json.dumps(idea))
                            print(f"Proposal finalized: {idea}")
                            shared_transcript.append(
                                f"[{agent_name} round {reflection_round + 1}] "
                                f"Proposal finalized: {trim_for_transcript(idea)}"
                            )
                            idea_finalized = True
                            break
                        except json.JSONDecodeError:
                            raise ValueError("Invalid arguments JSON for FinalizeIdea.")
                    else:
                        print(
                            "Invalid action. Please specify one of the available tools."
                        )
                        print(f"Available actions are: {tool_names_str}")
                except Exception as e:
                    print(
                        f"Failed to parse LLM response. Response text:\n{response_text}"
                    )
                    traceback.print_exc()
                    agent_histories[agent_name].append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response could not be parsed. Respond "
                                "using exactly this format:\n\nACTION:\n"
                                "<one available action>\n\nARGUMENTS:\n"
                                "{\"query\": \"[BROAD] ...\"} for search, or "
                                "{\"idea\": {...}} for FinalizeIdea. Do not include "
                                "markdown fences or extra text."
                            ),
                        }
                    )
                    last_tool_results = "Previous response was unparsable; retry with the required ACTION/ARGUMENTS format."
                    shared_transcript.append(
                        f"[{agent_name} round {reflection_round + 1}] "
                        f"Parse failure; Response={trim_for_transcript(response_text)}"
                    )
                    continue

            if idea_finalized:
                continue  # Move to the next idea

        except Exception as e:
            print("Failed to generate proposal:")
            traceback.print_exc()
            continue

    # Save ideas
    ideas = [json.loads(idea_str) for idea_str in idea_str_archive]

    with open(idea_fname, "w") as f:
        json.dump(ideas, f, indent=4)
    print(f"Stored {len(ideas)} ideas in {idea_fname}")
    return ideas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI scientist proposals - template free"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--max-num-generations",
        type=int,
        default=1,
        help="Maximum number of proposal generations.",
    )
    parser.add_argument(
        "--workshop-file",
        type=str,
        default="ideas/i_cant_believe_its_not_better.md",
        help="Path to the workshop description file.",
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=5,
        help=(
            "Number of reflection rounds per proposal in single mode, or full "
            "multi-agent cycles per proposal in multi mode."
        ),
    )
    parser.add_argument(
        "--min-num-searches",
        type=int,
        default=4,
        help=(
            "Minimum number of successful literature searches required before "
            "finalizing each proposal. Searches with no results, unavailable "
            "API keys, or errors do not count."
        ),
    )
    parser.add_argument(
        "--min-num-search-sources",
        type=int,
        default=3,
        help=(
            "Minimum number of distinct successful search sources required "
            "before finalizing each proposal."
        ),
    )
    parser.add_argument(
        "--min-num-search-purposes",
        type=int,
        default=3,
        help=(
            "Minimum number of distinct successful search purpose tags required "
            "before finalizing each proposal. Purpose tags are [BROAD], "
            "[OVERLAP], [DISCONFIRM], and [DATASET]."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Optional max output tokens per LLM call. Useful for local thinking "
            "models that need more room for reasoning plus final JSON."
        ),
    )
    parser.add_argument(
        "--agent-mode",
        type=str,
        default="single",
        choices=["single", "multi"],
        help=(
            "Ideation agent mode. 'single' preserves the original one-agent "
            "reflection loop; 'multi' rotates Explorer, Skeptic, Experiment, "
            "and Synthesizer agents with a shared transcript."
        ),
    )
    args = parser.parse_args()
    if args.min_num_search_sources > args.min_num_searches:
        parser.error(
            "--min-num-search-sources cannot be greater than --min-num-searches."
        )
    if args.min_num_search_purposes > len(SEARCH_PURPOSES):
        parser.error(
            f"--min-num-search-purposes cannot be greater than {len(SEARCH_PURPOSES)}."
        )
    if args.min_num_search_purposes > args.min_num_searches:
        parser.error(
            "--min-num-search-purposes cannot be greater than --min-num-searches."
        )
    total_planned_agent_rounds = args.num_reflections * len(
        get_agent_schedule(args.agent_mode)
    )
    if total_planned_agent_rounds <= args.min_num_searches:
        parser.error(
            "The planned number of agent calls must be greater than "
            "--min-num-searches so there is at least one round left to finalize "
            "the proposal. In multi mode, planned calls are "
            "--num-reflections multiplied by the number of agents."
        )

    # Create the LLM client
    client, client_model = create_client(args.model)

    with open(args.workshop_file, "r") as f:
        workshop_description = f.read()
    print(f"Using workshop description from {args.workshop_file} for idea generation.")
    print(f"Workshop description:\n{workshop_description}")

    # Create output filename by replacing .md extension with .json
    idea_fname = args.workshop_file.replace(".md", ".json")
    print("Starting idea generation for", idea_fname)
    ideas = generate_temp_free_idea(
        idea_fname=idea_fname,
        client=client,
        model=client_model,
        workshop_description=workshop_description,
        max_num_generations=args.max_num_generations,
        num_reflections=args.num_reflections,
        min_num_searches=args.min_num_searches,
        min_num_search_sources=args.min_num_search_sources,
        min_num_search_purposes=args.min_num_search_purposes,
        max_tokens=args.max_tokens,
        agent_mode=args.agent_mode,
    )
    print(f"{args.workshop_file} generated {len(ideas)} ideas.")
