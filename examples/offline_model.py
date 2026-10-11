"""Offline stand-in for a chat model, used by the demo notebook.

This is the same idea as the test fake: a ``BaseChatModel`` whose
``with_structured_output`` reads the prompt and returns the package schemas.
The rules below are keyword matches on this fictional survey, so the notebook
can run without an API key. They are not a grounded-theory analysis.
"""

from __future__ import annotations

import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from automated_grounded_theory.prompts import message_text
from automated_grounded_theory.retrieval import jaccard
from automated_grounded_theory.schemas import (
    BatchCoding,
    CodeProposal,
    DefinitionUpdate,
    MergeDecision,
    PassageCoding,
    SweepCoding,
    ThemeCoding,
    ThemeNegative,
    ThemeProposal,
    ThemeTension,
)

# First matching family wins for a passage, then a second family if one remains.
# Labels are gerunds. Two pay labels and two staffing labels are close on purpose
# so a sweep has a lexical pair to merge.
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hearing changes through rumors", ("rumor mill", "linkedin")),
    ("appreciating clearer updates", ("transparent", "town hall", "town halls", "more honest", "all hands", "gotten better", "walks the floor")),
    ("missing cross-department communication", ("between departments", "sales promises", "two companies")),
    ("doubting the survey will change anything", ("survey", "nobody reads", "slide that says")),
    ("fearing the next layoff", ("layoff", "layoffs", "restructure", "reorg", "job security", "walked out")),
    ("weighing inclusion against the talk", ("inclusion", "dei", "woman", "diverse", "erg")),
    ("asking for mental health support", ("mental health", "eap")),
    ("asking leaders to own mistakes", ("own mistakes", "blaming frontline")),
    ("calling reviews a box-checking exercise", ("performance review", "box checking")),
    ("wanting a say in decisions", ("more say", "approval layer", "signatures")),
    ("feeling like a number", ("like a number", "acquisition", "growing pains")),
    ("asking for clearer strategy", ("strategy", "where are we going")),
    ("comparing this place favorably", ("good place to work", "better than most", "good company")),
    ("having a manager who listens", ("listens", "listen", "has my back", "stretch project", "real feedback")),
    ("pushing back on micromanaging", ("micromanag",)),
    ("needing manager training", ("training on how to manage", "best picker")),
    ("missing a promotion path", ("promot", "career path", "don't see a path", "dont see a path", "hiring outside")),
    ("using chances to learn", ("mentor", "tuition", "learning", "cross training", "onboarding", "training", "course", "lead a team", "growth")),
    ("wanting recognition for the work", ("recogni", "thank you", "undervalued", "gift card", "credit", "reward")),
    ("valuing schedule flexibility", ("work from home", "hybrid", "remote", "flexib", "rto", "4 day", "start time", "schedule change", "work life", "balance")),
    ("running short of staff", ("burnout", "exhausted", "drowning", "stretched", "overtime", "do more with less", "tired", "fires")),
    ("running short on staff", ("staffing", "backfill", "running short", "6 days", "hire more")),
    ("wanting more pay", ("raise", "raises", "inflation")),
    (
        "wanting better pay",
        ("below market", "pay", "compensation", "paycheck", "bonus", "differential"),
    ),
    ("weighing the benefits package", ("benefit", "insurance", "deductible", "pto", "401k", "parental")),
    ("fighting clunky systems", ("tms", "wms", "spreadsheet", "intranet", "system", "laptop", "tools", "carriers", "red tape")),
    ("asking for a workplace fix", ("hvac", "parking", "microwave", "cafeteria", "forklift", "equipment", "break room")),
    ("sitting through extra meetings", ("meeting",)),
    ("keeping autonomy in the work", ("autonomy", "nobody micromanages")),
    ("feeling safe at work", ("safety", "safe at work")),
    ("seeing the point of the work", ("mission", "customer", "supply chain", "impact of my work", "dashboards", "solve real problems", "interesting work", "automation")),
    ("staying for the team", ("my team", "the people", "coworkers", "direct team", "pitches in", "collaborat", "smart people", "shoutout", "teamwork", "ops team", "recommend")),
    ("naming a small perk", ("free coffee", "bbq", "volunteer", "close to home")),
    ("finding little to like", ("nothing really", "not much", "its a job", "it's a job", "nothing comes to mind", "meh")),
)

_SKIP = frozenset(
    {
        "nope",
        "n/a",
        "na",
        "none",
        "idk",
        "dont know",
        "don't know",
        "thanks for asking",
        "glad to be here",
        "overall satisfied",
        "good luck with the results",
        "nothing, its fine",
        "nothing, it's fine",
    }
)

_THEME_OF = (
    ("pay", "Pay and benefits"),
    ("benefit", "Pay and benefits"),
    ("staff", "Workload and staffing"),
    ("flexib", "Schedule flexibility"),
    ("schedule", "Schedule flexibility"),
    ("rumor", "Leadership and trust"),
    ("update", "Leadership and trust"),
    ("communication", "Leadership and trust"),
    ("manager", "Leadership and trust"),
    ("micromanag", "Leadership and trust"),
    ("survey", "Leadership and trust"),
    ("mental", "Workload and staffing"),
    ("mistake", "Leadership and trust"),
    ("review", "Growth and recognition"),
    ("say in", "Leadership and trust"),
    ("number", "Leadership and trust"),
    ("strategy", "Leadership and trust"),
    ("favorably", "The work and the workplace"),
    ("layoff", "Leadership and trust"),
    ("inclusion", "Leadership and trust"),
    ("promotion", "Growth and recognition"),
    ("learn", "Growth and recognition"),
    ("recognition", "Growth and recognition"),
    ("system", "The work and the workplace"),
    ("workplace", "The work and the workplace"),
    ("meeting", "The work and the workplace"),
    ("team", "The work and the workplace"),
    ("safe", "The work and the workplace"),
    ("autonomy", "The work and the workplace"),
    ("point of the work", "The work and the workplace"),
    ("perk", "The work and the workplace"),
    ("little to like", "The work and the workplace"),
)

_THEME_COPY = {
    "Pay and benefits": (
        "Incidents about pay, raises, and the benefits package.",
        ["level of pay", "benefits versus take-home pay"],
    ),
    "Workload and staffing": (
        "Incidents about short staffing, hours, and exhaustion.",
        ["how long the shortage lasts", "who absorbs the work"],
    ),
    "Schedule flexibility": (
        "Incidents about hybrid work, remote days, and who gets that flexibility.",
        ["who is offered it", "how stable the policy is"],
    ),
    "Leadership and trust": (
        "Incidents about managers, communication, layoffs, and whether people trust what they hear.",
        ["how news arrives", "whether talk matches action"],
    ),
    "Growth and recognition": (
        "Incidents about promotion, learning, and being noticed for the work.",
        ["whether a path is visible", "what recognition is worth"],
    ),
    "The work and the workplace": (
        "Incidents about tools, the building, coworkers, and what the work is for.",
        ["tools versus people", "how much the work itself matters"],
    ),
    "Other incidents": (
        "Incidents that do not sit with the other themes.",
        ["leftover"],
    ),
}


def _passage_blocks(text: str) -> list[tuple[str, str]]:
    ids = re.findall(r"^passage_id: (.+)$", text, re.M)
    blocks = re.findall(r"<<<PASSAGE\n(.*?)\nPASSAGE>>>", text, re.S)
    pairs = []
    for index, block in enumerate(blocks):
        passage_id = ids[index] if index < len(ids) else ""
        pairs.append((passage_id, block))
    return pairs


def _body(passage: str) -> str:
    match = re.match(r"^(?:Like|Improve|Other):\s*", passage)
    return passage[match.end() :] if match else passage


def _quote(passage: str, needle: str) -> str:
    lowered = passage.casefold()
    index = lowered.find(needle.casefold())
    if index < 0:
        return passage.strip()
    start = max(passage.rfind(".", 0, index), passage.rfind("?", 0, index), passage.rfind("!", 0, index))
    start = 0 if start < 0 else start + 1
    end = len(passage)
    for punct in ".!?":
        found = passage.find(punct, index + len(needle))
        if found != -1:
            end = min(end, found + 1)
    snippet = passage[start:end].strip()
    comment = re.sub(r"^(?:Like|Improve|Other):\s*", "", snippet).strip()
    if comment and comment in passage:
        return comment
    return snippet or passage.strip()


def _matches(passage: str) -> list[tuple[str, str]]:
    body = _body(passage).strip()
    folded = re.sub(r"\s+", " ", body).casefold().strip(" .!?\"'")
    if not folded or folded in _SKIP:
        return []
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, needles in _RULES:
        if label in seen:
            continue
        for needle in needles:
            if needle.casefold() in folded:
                found.append((label, needle))
                seen.add(label)
                break
        if len(found) == 2:
            break
    return found


def _labels(text: str) -> list[str]:
    return re.findall(r"^CODE_LABEL: (.+)$", text, re.M)


def _theme_name(label: str) -> str:
    lowered = label.casefold()
    for needle, name in _THEME_OF:
        if needle in lowered:
            return name
    return "Other incidents"


class SurveyCoder(BaseChatModel):
    """Keyword coder for the fictional Northwind survey. No network calls."""

    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "offline-survey-coder"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            text = "\n".join(message_text(message) for message in messages)
            name = getattr(schema, "__name__", "")
            if name == "SweepCoding":
                parsed = self.sweep(text)
            elif name == "BatchCoding":
                parsed = self.batch(text)
            else:
                parsed = self.code(text)
            if not include_raw:
                return parsed
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 12, "output_tokens": 6, "total_tokens": 18},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)

    def batch(self, text: str) -> BatchCoding:
        header = text.split("PASSAGE TO CODE", 1)[0]
        feedback = ""
        if "VALIDATION FEEDBACK" in text:
            feedback = text[text.index("VALIDATION FEEDBACK") :]
        items = []
        for passage_id, block in _passage_blocks(text):
            one = (
                header
                + "\nPASSAGE TO CODE\n"
                + f"passage_id: {passage_id}\n"
                + "<<<PASSAGE\n"
                + block
                + "\nPASSAGE>>>\n"
                + feedback
            )
            coded = self.code(one)
            coded.passage_id = passage_id
            items.append(coded)
        if not items:
            items.append(self.code(text))
        return BatchCoding(passages=items)

    def code(self, text: str) -> PassageCoding:
        blocks = _passage_blocks(text)
        passage = blocks[0][1] if blocks else ""
        hits = _matches(passage)
        if not hits:
            return PassageCoding(codes=[], memo="")
        known = {item.casefold() for item in _labels(text)}
        others = [item for item in _labels(text)]
        proposals = []
        for label, needle in hits:
            action = "reuse" if label.casefold() in known else "new"
            nearest = ""
            why = ""
            if action == "new" and others:
                nearest = next((item for item in others if item.casefold() != label.casefold()), others[0])
                why = "It names a different incident."
            proposals.append(
                CodeProposal(
                    action=action,
                    code=label,
                    nearest_code=nearest,
                    why_not_nearest=why,
                    definition=f"Incidents of {label}.",
                    quote=_quote(passage, needle),
                    in_vivo=False,
                    rationale="The wording names this incident.",
                    confidence=0.9,
                )
            )
        return PassageCoding(
            codes=proposals,
            memo="Compared with the codebook: this passage adds a condition the nearest code does not cover.",
        )

    def sweep(self, text: str) -> SweepCoding:
        match = re.search(r"<<<CANDIDATES\n(.*)\nCANDIDATES>>>", text, re.S)
        pairs = json.loads(match.group(1)) if match else []
        merges = []
        for pair in pairs:
            source, target = pair["source"], pair["target"]
            if jaccard(source, target) < 0.5:
                continue
            merges.append(
                MergeDecision(
                    source=source,
                    target=target,
                    rationale="These labels name the same incident.",
                )
            )
        present = {label.casefold() for label in _labels(text)}
        definitions = []
        for label, definition in (
            ("wanting better pay", "Describing pay as below what the work is worth."),
            ("running short on staff", "Describing too few people for the work that remains."),
        ):
            if label.casefold() in present:
                definitions.append(DefinitionUpdate(code=label, definition=definition))
        return SweepCoding(
            merges=merges,
            definitions=definitions,
            integrative_memo="Constant comparison folded labels that name one incident.",
            reflexive_memo="These codes came from keyword rules on fictional comments. A researcher judges saturation.",
        )


class SurveyThemer(BaseChatModel):
    """Groups survey open-code labels into a few themes. No network calls."""

    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "offline-survey-themer"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            text = "\n".join(message_text(message) for message in messages)
            parsed = self.themes(text)
            if not include_raw:
                return parsed
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)

    def themes(self, text: str) -> ThemeCoding:
        labels = re.findall(r"^CODE_LABEL: (.+)$", text, re.M)
        groups: dict[str, list[str]] = {}
        for label in labels:
            groups.setdefault(_theme_name(label), []).append(label)
        themes = []
        for name, codes in groups.items():
            definition, properties = _THEME_COPY.get(name, _THEME_COPY["Other incidents"])
            themes.append(
                ThemeProposal(
                    name=name,
                    definition=definition,
                    properties=list(properties),
                    codes=codes,
                )
            )
        example = ""
        match = re.search(r"^EXAMPLE: (.+)$", text, re.M)
        if match and "…" not in match.group(1):
            example = match.group(1).strip()
        passage_id = ""
        id_match = re.search(r"^PASSAGE_ID: (.+)$", text, re.M)
        if id_match:
            passage_id = id_match.group(1).strip()
        names = [theme.name for theme in themes]
        tensions = []
        if len(names) >= 2:
            tensions.append(
                ThemeTension(
                    themes=names[:2],
                    note="These themes pull in different directions inside the same workplace.",
                    quote=example,
                    passage_id=passage_id if example else "",
                )
            )
        negatives = []
        if themes:
            negatives.append(
                ThemeNegative(
                    theme=themes[0].name,
                    note="A thin reply sits awkwardly beside richer incidents in this theme.",
                    code=themes[0].codes[0],
                    quote="",
                )
            )
        return ThemeCoding(themes=themes, tensions=tensions, negative_cases=negatives)
