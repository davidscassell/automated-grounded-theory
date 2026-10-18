"""0.9.15 regressions. Fake models only. No live evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import Field, ValidationError

from automated_grounded_theory import axial_code, selective_code
from automated_grounded_theory.async_support import CallLimiter, ainvoke_structured, run_sync
from automated_grounded_theory.axial import AxialProposalBatch, AxialRelationshipDraft, export_axial_state
from automated_grounded_theory.graph_grounding import graph_supported
from tests.fake_focused import FakeFocusedCoder
from tests.test_v080 import QUOTE, FakeAxialCoder, _corpus, _payload
from tests.test_v090 import FakeSamplingCoder


_COMMON = dict(
    seed=3,
    max_axial_review_calls=2,
    axial_share_propose=1,
    axial_share_counterexample=0,
    batch_size=1,
    interrupt=False,
)


def _texts(messages) -> str:
    parts = []
    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


class _RepairCoder(FakeAxialCoder):
    """First proposal is raw JSON with a blank statement. The repair is one call."""

    outcome: str = "verbatim"

    def with_structured_output(self, schema, **kwargs):
        def _call(messages):
            name = getattr(schema, "__name__", "")
            if name != "AxialProposalBatch":
                raise AssertionError(name)
            self.call_count += 1
            body = _payload(messages)
            pair = (body.get("pairs") or [{}])[0]
            from_id = str(pair.get("from_id") or "c-prep")
            to_id = str(pair.get("to_id") or "c-return")
            if "REPAIR THIS OUTPUT" not in _texts(messages):
                raw = AIMessage(
                    content=json.dumps(
                        {
                            "relationships": [
                                {
                                    "from_id": from_id,
                                    "to_id": to_id,
                                    "statement": "   ",
                                    "passage_id": "p1",
                                    "quote": QUOTE,
                                }
                            ]
                        }
                    )
                )
                return {"raw": raw, "parsed": None, "parsing_error": ValueError("blank statement")}
            if self.outcome == "still_blank":
                raw = AIMessage(
                    content=json.dumps(
                        {
                            "relationships": [
                                {
                                    "from_id": from_id,
                                    "to_id": to_id,
                                    "statement": "",
                                    "passage_id": "p1",
                                    "quote": QUOTE,
                                }
                            ]
                        }
                    )
                )
                return {"raw": raw, "parsed": None, "parsing_error": ValueError("blank statement")}
            if self.outcome == "explicit_none":
                parsed = AxialProposalBatch(relationships=[], nothing_to_propose=True)
            elif self.outcome == "verbatim":
                parsed = AxialProposalBatch(
                    relationships=[
                        AxialRelationshipDraft(
                            from_id=from_id,
                            to_id=to_id,
                            statement=QUOTE,
                            passage_id="p1",
                            quote=QUOTE,
                        )
                    ]
                )
            elif self.outcome == "stitched":
                parsed = AxialProposalBatch(
                    relationships=[
                        AxialRelationshipDraft(
                            from_id=from_id,
                            to_id=to_id,
                            statement="Packing may be related to returning.",
                            passage_id="p1",
                            quote_from="Packing the suitcase took all night",
                            quote_to="The return to work after leave",
                        )
                    ]
                )
            elif self.outcome == "unanchored":
                parsed = AxialProposalBatch(
                    relationships=[
                        AxialRelationshipDraft(
                            from_id=from_id,
                            to_id=to_id,
                            statement="The schedule may be related to the meeting.",
                            passage_id="p1",
                            quote="this quotation is not in the passage",
                        )
                    ]
                )
            else:
                parsed = AxialProposalBatch(
                    relationships=[
                        AxialRelationshipDraft(
                            from_id=from_id,
                            to_id=to_id,
                            statement="Packing may be related to the trip.",
                            passage_id="p1",
                            quote=QUOTE,
                        )
                    ]
                )
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def _repair_row(result):
    rows = [row for row in result.prompt_log if row.get("kind") == "schema_repair"]
    assert len(rows) == 1
    blob = json.dumps(rows[0]["messages"])
    assert "REPAIR THIS OUTPUT" in blob
    assert "Packing the suitcase took all night before the trip." in blob
    return rows[0]


def test_statement_schema_rejects_blanks_and_one_repair_stays_grounded():
    """A blank statement fails the schema. One repair is logged against the original evidence.

    A recovered statement that is only a quote is not a relationship. A recovered
    claim still passes the passage and stitching checks.
    """
    with pytest.raises(ValidationError):
        AxialRelationshipDraft()
    with pytest.raises(ValidationError):
        AxialRelationshipDraft(statement="   ", from_id="c-prep", to_id="c-return")
    empty = AxialProposalBatch(nothing_to_propose=True, relationships=[])
    assert empty.nothing_to_propose is True
    assert empty.relationships == []

    verbatim = _RepairCoder(outcome="verbatim")
    quoted = axial_code(_corpus(), verbatim, **_COMMON)
    assert verbatim.call_count == 2
    assert quoted.relationships == []
    assert quoted.metadata["proposal_diagnostics"]["schema_repairs"] == 1
    assert quoted.metadata["proposal_diagnostics"]["rejection_reasons"]["verbatim_quote_statement"] == 1
    _repair_row(quoted)

    grounded = _RepairCoder(outcome="grounded")
    kept = axial_code(_corpus(), grounded, **_COMMON)
    assert grounded.call_count == 2
    assert len(kept.relationships) == 1
    assert kept.relationships[0]["statement"] == "Packing may be related to the trip."
    assert kept.relationships[0]["endpoint_anchoring"]
    assert "verbatim_quote_statement" not in kept.metadata["proposal_diagnostics"]["rejection_reasons"]

    stitched = _RepairCoder(outcome="stitched")
    dropped = axial_code(_corpus(), stitched, **_COMMON)
    assert stitched.call_count == 2
    assert dropped.relationships == []
    assert dropped.metadata["proposal_diagnostics"]["rejection_reasons"]["stitched_unrelated_passages"] == 1

    loose = _RepairCoder(outcome="unanchored")
    held = axial_code(_corpus(), loose, **_COMMON)
    assert len(held.relationships) == 1
    assert held.relationships[0]["evidence_condition"] == "unanchored"
    assert held.relationships[0]["downstream_eligible"] is False

    explicit = _RepairCoder(outcome="explicit_none")
    none = axial_code(_corpus(), explicit, **_COMMON)
    assert explicit.call_count == 2
    assert none.relationships == []
    assert none.metadata["proposal_diagnostics"]["explicit_none"] == 1

    failed = _RepairCoder(outcome="still_blank")
    unparsed = axial_code(_corpus(), failed, **_COMMON)
    assert failed.call_count == 2
    assert unparsed.relationships == []
    assert unparsed.metadata["proposal_diagnostics"]["schema_repairs"] == 1
    assert unparsed.metadata.get("axial_stop") == "schema_unparsed"
    repair_rows = [row for row in unparsed.prompt_log if row.get("kind") == "schema_repair"]
    assert len(repair_rows) == 1
    assert repair_rows[0]["status"] == "schema_unparsed"


def test_weak_stem_cannot_reach_supported_evidence_consumers(tmp_path: Path):
    """A same-passage stem is hypothesis exploration, even if eligibility was set true.

    Core graph support ignores that stale flag. Export rewrites it. A memo and
    a storyline still receive the row as a hypothesis, not as supported evidence.
    A tier-2 pattern without a weak stem still supports the graph. Causal
    wording on that pattern withholds eligibility and does not remove graph support.
    """
    quote = "the people in my shift. management not so much"
    corpus = {
        "codebook": {
            "c-shift": {"code_id": "c-shift", "code": "shift", "definition": "shift"},
            "c-mgmt": {"code_id": "c-mgmt", "code": "management", "definition": "management"},
        },
        "passages": [{"passage_id": "p-shift", "doc_id": "d1", "respondent_id": "r1", "text": quote}],
        "segments": [
            {"segment_id": "s1", "code_id": "c-shift", "passage_id": "p-shift"},
            {"segment_id": "s2", "code_id": "c-mgmt", "passage_id": "p-shift"},
        ],
    }

    class _Stem(FakeAxialCoder):
        def proposals(self, body):
            pair = (body.get("pairs") or [{}])[0]
            return AxialProposalBatch(
                relationships=[
                    AxialRelationshipDraft(
                        from_id=str(pair.get("from_id") or "c-shift"),
                        to_id=str(pair.get("to_id") or "c-mgmt"),
                        paradigm_role="context",
                        statement="Shift may be related to management.",
                        passage_id="p-shift",
                        quote=quote,
                    )
                ]
            )

    coded = axial_code(
        corpus,
        _Stem(),
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )
    row = coded.relationships[0]
    assert row["evidence_scope"] == "same_passage_stem_only"
    assert "weak_stem_anchor" in row["flags"]
    assert row["evidence_use"] == "hypothesis_exploration"
    assert row["downstream_eligible"] is False

    poisoned = dict(row)
    poisoned["evidence_tier"] = 2
    poisoned["evidence_label"] = "pattern_supported"
    poisoned["evidence_status"] = "pattern_supported"
    poisoned["evidence_condition"] = "anchored"
    poisoned["disposition"] = "accepted"
    poisoned["downstream_eligible"] = True
    assert graph_supported(poisoned) is False

    paths = export_axial_state(
        {"relationships": [poisoned], "metadata": {"stage": "axial"}, "codebook": corpus["codebook"]},
        tmp_path,
    )
    saved = json.loads(Path(paths["category_relationships.json"]).read_text(encoding="utf-8"))
    assert saved["relationships"][0]["downstream_eligible"] is False
    assert saved["relationships"][0]["evidence_use"] == "hypothesis_exploration"

    coded.relationships = [poisoned]
    coded.codebook = [
        {"code_id": "c-shift", "code": "shift", "definition": "shift"},
        {"code_id": "c-mgmt", "code": "management", "definition": "management"},
    ]
    coded.passages = list(corpus["passages"])
    coded.segments = list(corpus["segments"])
    chosen = selective_code(
        coded,
        FakeAxialCoder(core_id="c-shift"),
        decision_mode="llm",
        include_memos=True,
        include_storyline=True,
        interrupt=False,
        max_selective_review_calls=4,
    )
    assert chosen.metadata["core_unsupported_by_graph"] is True
    assert chosen.relationships[0]["downstream_eligible"] is False
    assert chosen.relationships[0]["evidence_use"] == "hypothesis_exploration"
    memo_prompt = next(item for item in chosen.prompt_log if item.get("kind") == "theoretical_memo")
    memo_body = json.loads(memo_prompt["messages"][-1]["content"])
    sent_ids = [item.get("relationship_id") for item in memo_body["relationships"]]
    assert row["relationship_id"] in sent_ids
    sent = next(item for item in memo_body["relationships"] if item.get("relationship_id") == row["relationship_id"])
    assert sent["downstream_eligible"] is False
    assert sent["evidence_use"] == "hypothesis_exploration"
    assert chosen.metadata["graph_snapshots"]["storyline"]["relationship_ids"] == [row["relationship_id"]]
    assert chosen.metadata["storyline_core_emphasis"] is False

    strong = {
        "relationship_id": "rel-strong",
        "from_id": "a",
        "to_id": "b",
        "statement": "The weekly meeting may be related to the written agenda.",
        "evidence_tier": 2,
        "evidence_label": "pattern_supported",
        "evidence_condition": "anchored",
        "disposition": "accepted",
        "downstream_eligible": False,
        "endpoint_anchoring": [
            {"endpoint": "from", "anchored": True, "reason": ""},
            {"endpoint": "to", "anchored": True, "reason": ""},
        ],
        "flags": [],
    }
    assert graph_supported(strong) is True
    causal = dict(strong)
    causal["relationship_id"] = "rel-causal"
    causal["flags"] = ["causal_wording_withheld", "wording_exceeds_tier"]
    assert graph_supported(causal) is True
    causal_paths = export_axial_state(
        {"relationships": [strong, causal], "metadata": {"stage": "axial"}, "codebook": {}},
        tmp_path / "strong",
    )
    exported = json.loads(Path(causal_paths["category_relationships.json"]).read_text(encoding="utf-8"))
    by_id = {item["relationship_id"]: item for item in exported["relationships"]}
    assert by_id["rel-strong"]["evidence_use"] == "supported_evidence"
    assert by_id["rel-strong"]["downstream_eligible"] is True
    assert by_id["rel-causal"]["evidence_use"] == "withheld"
    assert by_id["rel-causal"]["downstream_eligible"] is False


def test_association_may_stay_unspecified_and_role_evidence_is_not_statement_wording():
    """A causal role below tier 3 is still withheld. The sentence is a separate flag.

    unspecified is allowed when the statement is only an association.
    """
    from automated_grounded_theory.axial import _KIND_PROMPTS
    from automated_grounded_theory.gate_evaluation import export_gate_relationships
    from automated_grounded_theory.relationship_tiers import role_evidence_overreach, wording_overreach

    prompt = _KIND_PROMPTS["axial_proposal"]
    assert "only an association" in prompt
    assert "unspecified is not the default" not in prompt
    assert wording_overreach("The teams met weekly.", 2, role="consequence") is True
    assert wording_overreach("leads to", 2) is True
    assert role_evidence_overreach(2, "consequence") is True
    assert role_evidence_overreach(3, "consequence") is False

    text = "The tools for the shift sit beside coworkers on the crew tonight"
    corpus = {
        "codebook": {
            "tools": {"code_id": "tools", "code": "tools for the shift", "definition": "tools for the shift"},
            "crew": {"code_id": "crew", "code": "coworkers on the crew", "definition": "coworkers on the crew"},
        },
        "passages": [{"passage_id": "p1", "doc_id": "d1", "respondent_id": "r1", "text": text}],
        "segments": [
            {"segment_id": "s1", "code_id": "tools", "passage_id": "p1"},
            {"segment_id": "s2", "code_id": "crew", "passage_id": "p1"},
        ],
    }

    class _Roles(FakeAxialCoder):
        def proposals(self, body):
            pair = (body.get("pairs") or [{}])[0]
            from_id = str(pair.get("from_id") or "tools")
            to_id = str(pair.get("to_id") or "crew")
            shared = dict(
                from_id=from_id,
                to_id=to_id,
                passage_id="p1",
                quote_from="tools for the shift sit",
                quote_to="coworkers on the crew tonight",
            )
            return AxialProposalBatch(
                relationships=[
                    AxialRelationshipDraft(
                        paradigm_role="unspecified",
                        statement="The tools may be related to the crew.",
                        **shared,
                    ),
                    AxialRelationshipDraft(
                        paradigm_role="consequence",
                        statement="The teams met weekly.",
                        **shared,
                    ),
                    AxialRelationshipDraft(
                        paradigm_role="context",
                        statement="The tools lead to the crew.",
                        **shared,
                    ),
                ]
            )

    result = axial_code(
        corpus,
        _Roles(),
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )
    by_statement = {row["statement"]: row for row in result.relationships}
    association = by_statement["The tools may be related to the crew."]
    assert association["paradigm_role"] == "unspecified"
    assert "paradigm_role_unspecified" not in association["flags"]
    assert "role_evidence_exceeds_tier" not in association["flags"]
    assert "wording_exceeds_tier" not in association["flags"]
    mild = by_statement["The teams met weekly."]
    assert "role_evidence_exceeds_tier" in mild["flags"]
    assert "wording_exceeds_tier" not in mild["flags"]
    assert "causal_wording_withheld" in mild["flags"]
    assert mild["downstream_eligible"] is False
    causal = by_statement["The tools lead to the crew."]
    assert "wording_exceeds_tier" in causal["flags"]
    assert "role_evidence_exceeds_tier" not in causal["flags"]
    assert "causal_wording_withheld" in causal["flags"]
    gate = export_gate_relationships(result.relationships)
    rules = {
        item["relationship_id"]: set(item.get("rules") or [])
        for bucket in ("accepted", "withheld", "proposed")
        for item in gate[bucket]
    }
    assert "role_evidence" in rules[mild["relationship_id"]]
    assert "consequence_wording" not in rules[mild["relationship_id"]]
    assert "consequence_wording" in rules[causal["relationship_id"]]
    assert "role_evidence" not in rules[causal["relationship_id"]]


def test_axial_export_records_attrition_and_refuses_a_version_mismatch(tmp_path: Path, monkeypatch):
    """The funnel deduplicates compared runs. Temperature stays unset unless configured."""
    import automated_grounded_theory as package
    from automated_grounded_theory.backends import InstallationMismatch, installation_record

    result = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )
    provenance = result.metadata["provenance"]
    assert provenance["module_version"] == package.__version__
    assert provenance["distribution_version"] == package.__version__
    assert provenance["module_path"].endswith("automated_grounded_theory/__init__.py")
    assert provenance["commit"]
    parameters = result.metadata["generation_parameters"]
    assert parameters["temperature"] is None
    assert parameters["temperature_status"] == "unset"
    assert parameters["model"]
    attrition = result.metadata["attrition"]
    assert attrition["unique_pairs"] == 1
    assert attrition["proposals"] is not None and attrition["proposals"] >= 1
    assert attrition["valid_statements"] == len(result.relationships)
    assert attrition["deduplicated_relationships"] is None
    assert "evidence_tiers" in attrition

    class _Hot(FakeAxialCoder):
        temperature: float = 0.2

    hot = axial_code(
        _corpus(),
        _Hot(),
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )
    assert hot.metadata["generation_parameters"]["temperature"] == 0.2
    assert hot.metadata["generation_parameters"]["temperature_status"] == "configured"

    repeated = dict(result.relationships[0])
    compared = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
        comparison_runs=[{"relationships": [repeated, dict(repeated)]}],
    )
    assert compared.metadata["attrition"]["deduplicated_relationships"] == len(
        {(row["from_id"], row["to_id"], row["statement"]) for row in compared.relationships + [repeated]}
    )
    paths = export_axial_state(
        {"relationships": compared.relationships, "metadata": compared.metadata, "codebook": {}},
        tmp_path,
    )
    saved = json.loads(Path(paths["axial_run.json"]).read_text(encoding="utf-8"))
    assert saved["metadata"]["attrition"]["deduplicated_relationships"] == compared.metadata["attrition"]["deduplicated_relationships"]
    assert saved["metadata"]["provenance"]["commit"] == installation_record()["commit"]

    def _other(name: str):
        del name
        raise AssertionError("distribution should be patched")

    monkeypatch.setattr(
        "automated_grounded_theory.backends.importlib.metadata.distribution",
        lambda name: type("Dist", (), {"version": "9.9.9", "locate_file": lambda self, rel: Path("/tmp/other-checkout")})(),
    )
    with pytest.raises(InstallationMismatch):
        installation_record()
    with pytest.raises(InstallationMismatch):
        export_axial_state({"relationships": [], "metadata": {}, "codebook": {}}, tmp_path / "mismatch")


class _StallCoder(FakeAxialCoder):
    """Sleep on the initial call, the repair call, or neither."""

    stall: str = ""
    delay: float = 1.5

    def with_structured_output(self, schema, **kwargs):
        parent = FakeAxialCoder.with_structured_output(self, schema, **kwargs)

        def _call(messages):
            name = getattr(schema, "__name__", "")
            repair = "REPAIR THIS OUTPUT" in _texts(messages)
            if name == "AxialProposalBatch" and self.stall == "initial" and not repair and self.call_count == 0:
                self.call_count += 1
                time.sleep(self.delay)
            if name == "AxialProposalBatch" and self.stall == "repair" and not repair:
                self.call_count += 1
                raw = AIMessage(content=json.dumps({"relationships": [{"statement": "   "}]}))
                return {"raw": raw, "parsed": None, "parsing_error": ValueError("blank statement")}
            if name == "AxialProposalBatch" and self.stall == "repair" and repair:
                self.call_count += 1
                time.sleep(self.delay)
            return parent.invoke(messages)

        return RunnableLambda(_call)


def test_stalled_initial_and_repair_calls_stay_inside_the_timeout(tmp_path: Path):
    """A stalled provider call is cut off. The failed initial response is kept.

    Resume after the initial timeout uses the same checkpoint and can finish.
    A stalled repair keeps the failed parse and a later call can succeed.
    """
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    timeout = 0.35
    initial = _StallCoder(stall="initial", delay=1.5)
    saver = InMemorySaver()
    started = time.perf_counter()
    partial = axial_code(
        _corpus(),
        initial,
        seed=3,
        max_axial_review_calls=2,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        axial_request_timeout=timeout,
        interrupt=False,
        checkpointer=saver,
        thread_id="stall-initial",
        output_dir=tmp_path / "initial",
    )
    assert time.perf_counter() - started < timeout + 0.9
    assert partial.metadata["stop"] == "timeout"
    assert partial.metadata["status"] == "incomplete"
    assert (tmp_path / "initial" / "axial_run.json").exists()
    assert "timed out after 0.35" in partial.metadata["timeout_error"]

    initial.stall = ""
    recovered = axial_code(
        _corpus(),
        initial,
        seed=3,
        max_axial_review_calls=2,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        axial_request_timeout=timeout,
        interrupt=False,
        checkpointer=saver,
        thread_id="stall-initial",
        resuming=True,
        resume={"accept": True},
    )
    assert recovered.relationships
    assert any(row.get("status") == "timeout" for row in recovered.prompt_log)

    repair = _StallCoder(stall="repair", delay=1.5)
    limiter = CallLimiter(2, max_retries=0, request_timeout=timeout, wall_timeout=timeout)
    audits: list[dict] = []
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match=r"0\.35"):
        run_sync(
            ainvoke_structured(
                repair,
                AxialProposalBatch,
                [HumanMessage(content='{"pairs":[{"from_id":"c-prep","to_id":"c-return"}]}')],
                limiter,
                preserve_failed=True,
                call_audits=audits,
            )
        )
    assert time.perf_counter() - started < timeout + 0.9
    assert audits[0]["status"] == "parse_failed"
    assert audits[0]["raw_response"]

    repair.stall = ""
    (recovered_payload, _usage), _messages = run_sync(
        ainvoke_structured(
            repair,
            AxialProposalBatch,
            [HumanMessage(content='{"pairs":[{"from_id":"c-prep","to_id":"c-return"}]}')],
            CallLimiter(2, max_retries=0, request_timeout=timeout, wall_timeout=timeout),
            preserve_failed=True,
        )
    )
    assert recovered_payload["relationships"]
    assert recovered_payload["relationships"][0]["statement"]


class _RereadReportCoder(FakeSamplingCoder):
    """Several reread judgments, including one pair that never parses."""

    fails: int = 0
    chosen: dict = Field(default_factory=dict)

    def with_structured_output(self, schema, **kwargs):
        include_raw = bool(kwargs.get("include_raw"))

        def _call(messages):
            self.call_count += 1
            name = getattr(schema, "__name__", "")
            body = _payload(messages)
            if name == "ModelSamplingSelection":
                kinds = {}
                for row in body.get("directions") or []:
                    kinds.setdefault(str(row.get("kind") or ""), str(row.get("direction_id") or ""))
                thin = kinds["thin_category"]
                other = kinds.get("no_negative_case") or thin
                self.chosen = {"thin": thin, "other": other}
                parsed = schema(
                    selections=[
                        {"direction_id": thin, "passage_id": "p1"},
                        {"direction_id": thin, "passage_id": "p-new"},
                        {"direction_id": other, "passage_id": "p2"},
                        {"direction_id": thin, "passage_id": "p-low"},
                        {"direction_id": thin, "passage_id": "p-none"},
                        {"direction_id": thin, "passage_id": "p-monday"},
                    ],
                    rationale="These pairs are the rereads. The flat lists were not sent.",
                )
            elif name == "SamplingRereadNote":
                passage_id = str(body.get("passage_id") or "")
                text = str(body.get("text") or "")
                if passage_id == "p-monday":
                    self.fails += 1
                    raw = AIMessage(content="not json")
                    return {"raw": raw, "parsed": None, "parsing_error": ValueError("not json")}
                if passage_id == "p2":
                    parsed = schema(passage_id=passage_id, judgment="irrelevant", note="Not this direction.", quote=text[:24])
                elif passage_id == "p-low":
                    parsed = schema(passage_id=passage_id, judgment="", note="No judgment was given.", quote=text[:24])
                elif passage_id == "p-none":
                    parsed = schema(
                        passage_id=passage_id,
                        judgment="relevant",
                        note="The quote is not in the passage.",
                        quote="this sentence was never written",
                    )
                else:
                    parsed = schema(passage_id=passage_id, judgment="relevant", note="This passage bears on the question.", quote=text[:24])
            elif name == "ModelSaturationJudgment":
                parsed = schema(
                    assessment="not_saturated",
                    rationale="The window is interpretable and a sampling gap is still open.",
                )
            else:
                parsed = schema()
            if not include_raw:
                return parsed
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_call)


def test_reread_reporting_splits_non_relevant_and_exposes_completed_pairs(tmp_path: Path):
    """Reporting only. The same notes are stored; the result names the pairs."""
    from automated_grounded_theory import theoretical_sample
    from tests.test_v090 import _kwargs

    corpus = _corpus()
    for passage in corpus["passages"]:
        if passage["passage_id"] == "p1":
            passage["respondent_id"] = "r-known"
    corpus["passages"].append(
        {
            "passage_id": "p-new",
            "doc_id": "d-new",
            "respondent_id": "r-new",
            "text": "A new ward opened after the night shift.",
        }
    )
    coder = _RereadReportCoder()
    result = theoretical_sample(
        corpus,
        coder,
        decision_mode="llm",
        pursuit="reread",
        interrupt=False,
        output_dir=tmp_path,
        **_kwargs(max_sampling_review_calls=8),
    )
    assert result.completed_pairs is not None
    assert result.completed_pairs == result.metadata["reread_completed_pairs"]
    assert {row["passage_id"] for row in result.completed_pairs} == {"p1", "p-new", "p2", "p-low", "p-none"}
    stored = {row["passage_id"]: row for row in result.reread_notes}
    assert stored["p-none"]["judgment"] == "relevant"
    assert "quote_unanchored" in stored["p-none"]["flags"]
    assert stored["p2"]["judgment"] == "irrelevant"
    report = result.metadata["reread_interpretations"]
    assert report["relevant"] == 2
    assert report["irrelevant"] == 1
    assert report["unjudged"] == 1
    assert report["unanchored"] == 1
    assert report["failed"] == 1
    assert report["non_relevant"] == 4
    assert report["relevance"] == round(2 / 6, 4)
    assert report["by_direction_type"]["thin_category"]["relevant"] == 2
    assert report["by_direction_type"]["thin_category"]["failed"] == 1
    other = coder.chosen["other"]
    other_kind = next(row["kind"] for row in result.directions if row["direction_id"] == other)
    assert report["by_direction_type"][other_kind]["irrelevant"] == 1
    assert report["by_document_novelty"]["new_document"]["relevant"] == 1
    assert report["by_document_novelty"]["known_document"]["relevant"] == 1
    assert report["by_respondent_novelty"]["new_respondent"]["relevant"] == 1
    assert report["by_respondent_novelty"]["known_respondent"]["relevant"] == 1
    assert "unresolved" not in report["by_document_novelty"]
    assert report["relevant_reread_tokens"] == 84
    assert report["tokens_per_relevant_reread"] == 42
    assert report["cost_per_relevant_reread_usd"] is None
    assert report["cost_status"] == "unset"
    saved = json.loads((tmp_path / "theoretical_sampling.json").read_text(encoding="utf-8"))
    assert saved["reread_interpretations"]["failed"] == 1
    assert saved["reread_completed_pairs"] == result.completed_pairs

    untouched = theoretical_sample(corpus, coder, decision_mode="llm", pursuit="brief_only", interrupt=False, **_kwargs(max_sampling_review_calls=2))
    assert untouched.completed_pairs is None
    assert untouched.metadata["reread_interpretations"]["relevance"] is None
    assert untouched.metadata["reread_interpretations"]["relevant_reread_tokens"] is None
    assert untouched.reread_notes == []


class _PreservedRepairCoder(_StallCoder):
    """Invalid first response carries usage. The repair stalls or crashes."""

    crash: bool = False

    def with_structured_output(self, schema, **kwargs):
        inner = _StallCoder.with_structured_output(self, schema, **kwargs)

        def _call(messages):
            repair = "REPAIR THIS OUTPUT" in _texts(messages)
            if repair and self.crash:
                raise RuntimeError("repair crashed")
            result = inner.invoke(messages)
            if isinstance(result, dict) and result.get("parsed") is None and isinstance(result.get("raw"), AIMessage):
                result["raw"].usage_metadata = {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
                result["raw"].content = json.dumps({"relationships": [{"statement": "   "}], "preserved": "original"})
            return result

        return RunnableLambda(_call)


def _original_usage(result) -> dict:
    hold = (result.metadata.get("provider_call_hold") or {})
    if hold.get("usage"):
        return hold["usage"]
    for row in result.prompt_log:
        usage = row.get("usage") or {}
        if usage.get("total_tokens") == 18:
            return usage
    return {}


def test_repair_stall_keeps_the_original_response_and_usage(tmp_path: Path):
    """A stall or crash during repair does not drop the first response."""
    from langgraph.checkpoint.memory import InMemorySaver

    timeout = 0.35
    stalled = _PreservedRepairCoder(stall="repair", delay=1.5)
    saver = InMemorySaver()
    started = time.perf_counter()
    partial = axial_code(
        _corpus(),
        stalled,
        axial_request_timeout=timeout,
        checkpointer=saver,
        thread_id="repair-stall",
        output_dir=tmp_path / "stall",
        **_COMMON,
    )
    assert time.perf_counter() - started < timeout + 0.9
    assert partial.metadata["stop"] == "timeout"
    hold = partial.metadata["provider_call_hold"]
    assert hold["kind"] == "axial_proposal"
    assert hold["usage"]["total_tokens"] == 18
    assert "preserved" in hold["raw_response"]
    assert _original_usage(partial)["input_tokens"] == 11
    logged = [row for row in partial.prompt_log if (row.get("usage") or {}).get("total_tokens") == 18]
    assert logged
    assert "preserved" in str(logged[0].get("raw_response") or "")

    crashed = _PreservedRepairCoder(stall="repair", crash=True)
    crash_saver = InMemorySaver()
    with pytest.raises(RuntimeError, match="repair crashed"):
        axial_code(
            _corpus(),
            crashed,
            axial_request_timeout=timeout,
            checkpointer=crash_saver,
            thread_id="repair-crash",
            **_COMMON,
        )
    saved = crash_saver.get_tuple({"configurable": {"thread_id": "repair-crash"}})
    values = saved.checkpoint["channel_values"]
    crashed_hold = values["metadata"]["provider_call_hold"]
    assert crashed_hold["usage"]["total_tokens"] == 18
    assert "preserved" in crashed_hold["raw_response"]
    assert any((row.get("usage") or {}).get("total_tokens") == 18 for row in values["prompt_log"])


class _FocusedRepairStall(FakeFocusedCoder):
    """The recode response is unparsed. The repair sleeps."""

    delay: float = 1.5

    def with_structured_output(self, schema, **kwargs):
        def _call(messages):
            self.call_count += 1
            text = "\n".join(getattr(message, "content", "") if not isinstance(message, dict) else str(message.get("content") or "") for message in messages)
            if "REPAIR THIS OUTPUT" in text:
                time.sleep(self.delay)
            raw = AIMessage(
                content="not json focused-original",
                usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            )
            return {"raw": raw, "parsed": None, "parsing_error": ValueError("not a recode")}

        return RunnableLambda(_call)


def test_focused_repair_stall_keeps_the_original_response(tmp_path: Path):
    from automated_grounded_theory import focused_code
    from tests.test_v070 import _entry, _passages, _source

    timeout = 0.35
    started = time.perf_counter()
    partial = focused_code(
        _source(
            passages=_passages()[:1],
            codes=[_entry("c-freq", "packing bags", 1, ["d1"])],
            segments=[
                {
                    "segment_id": "s1",
                    "code_id": "c-freq",
                    "code": "packing bags",
                    "passage_id": "p1",
                    "doc_id": "d1",
                    "quote": _passages()[0]["text"],
                    "review_reasons": [],
                }
            ],
        ),
        model=_FocusedRepairStall(),
        seed=1,
        focused_min_score=0,
        focused_neighbor_limit=0,
        focused_outside_limit=0,
        focused_negative_case_limit=0,
        focused_share_develop=1,
        focused_share_compete=0,
        focused_share_outside=0,
        focused_share_claim_compete=0,
        focused_request_timeout=timeout,
        max_focused_review_calls=2,
        interrupt=False,
        output_dir=tmp_path / "focused",
    )
    assert time.perf_counter() - started < timeout + 0.9
    assert partial.metadata["stop"] == "timeout"
    holds = partial.metadata["focused_call_holds"]
    preserved = next(iter(holds.values()))
    assert preserved["usage"]["total_tokens"] == 18
    assert "focused-original" in preserved["raw_response"]


class _RateCoder(FakeAxialCoder):
    def with_structured_output(self, schema, **kwargs):
        def _call(messages):
            self.call_count += 1
            error = RuntimeError("rate limit")
            error.status_code = 429
            raise error

        return RunnableLambda(_call)


class _BoomCoder(FakeAxialCoder):
    def with_structured_output(self, schema, **kwargs):
        def _call(messages):
            raise RuntimeError("provider exploded")

        return RunnableLambda(_call)


class _NullStatementCoder(FakeAxialCoder):
    seen: str = ""

    def with_structured_output(self, schema, **kwargs):
        parent = FakeAxialCoder.with_structured_output(self, schema, **kwargs)

        def _call(messages):
            text = _texts(messages)
            if getattr(schema, "__name__", "") == "AxialProposalBatch" and "REPAIR THIS OUTPUT" not in text:
                self.call_count += 1
                raw = AIMessage(content='{"relationships":[{"from_id":"c-prep","to_id":"c-return","statement":null}]}')
                return {"raw": raw, "parsed": None, "parsing_error": ValueError("statement is null")}
            self.seen = text
            return parent.invoke(messages)

        return RunnableLambda(_call)


def test_failure_routes_stay_separate():
    """Retries, repair, analytic outcomes, interrupts, and crashes stay apart.

    provider_max_attempts counts the initial call. The graph does not multiply
    it. The stage call allowance stops the retries first. A null statement is
    repair feedback. Missing category evidence is recorded, not raised. A
    missing researcher decision pauses with interrupt(). An unexpected error
    propagates.
    """
    limited = _RateCoder()
    stopped = axial_code(
        _corpus(),
        limited,
        seed=3,
        provider_max_attempts=4,
        retry_backoff=0,
        max_axial_review_calls=2,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=False,
    )
    assert limited.call_count == 2
    assert stopped.metadata["axial_stop"] == "call_allowance"
    assert stopped.metadata["retry_policy"] == {
        "provider_max_attempts": 4,
        "graph_max_attempts": 1,
        "counts_initial_attempt": True,
    }

    with pytest.raises(RuntimeError, match="provider exploded"):
        axial_code(_corpus(), _BoomCoder(), **_COMMON)

    repaired = _NullStatementCoder()
    axial_code(_corpus(), repaired, **_COMMON)
    assert "REPAIR THIS OUTPUT" in repaired.seen
    assert "statement" in repaired.seen.casefold()

    source = _corpus()
    source["segments"] = []
    deferred = axial_code(
        source,
        FakeAxialCoder(),
        seed=3,
        axial_token_budget=1,
        max_axial_review_calls=2,
        batch_size=1,
        interrupt=False,
    )
    assert deferred.metadata["missing_category_evidence"]
    assert any(row.get("reason") == "missing_category_evidence" and row.get("status") == "deferred" for row in deferred.prompt_log)

    paused = axial_code(
        _corpus(),
        FakeAxialCoder(),
        seed=3,
        max_axial_review_calls=1,
        axial_share_propose=1,
        axial_share_counterexample=0,
        batch_size=1,
        interrupt=True,
    )
    assert paused.interrupted is True
    assert paused.interrupt_payload["kind"] == "axial_relations"
