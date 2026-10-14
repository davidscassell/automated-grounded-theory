# Later-stage design

Status: focused coding and the property log (PR1) are implemented in package 0.7.0. Package 0.7.1 corrects selection ties, empty reread purposes, id counters loaded from an open-coding folder, confirmation versus a new span, the focused prompt, and the focused export. Package 0.7.2 sizes each reread queue to its reserved share, reports passages never queued, seeds competing readings before claims exist, shows the full codebook, flags near-duplicate codes and category mismatches, nests span proposals, and enforces a wall-clock request timeout. Package 0.7.3 reserves a claim-compete share, requires an explicit compete verdict and alternative reading, records provider token usage apart from the estimate, and reports property stability. Axial coding, selective coding, a core category, theoretical sampling, and a saturation decision in this note are not implemented. `judgment` stays null. This note was written against package 0.6.16 (`e1c7a74`).

The README and [methodology.md](methodology.md) describe focused coding as a separate graph. The theme report still does not reread passages, and it is not focused coding.

This document specifies how focused coding, axial coding, selective coding, theoretical saturation, and theoretical sampling inside a fixed corpus could be added. The default run remains initial coding plus a theme report. [methodology.md](methodology.md) states that limit. The engineering of the current run is [design.md](design.md).

The procedure schools do not agree on the order or the tools. The default proposed here follows Charmaz. A Strauss and Corbin paradigm, and Glaser's theoretical coding families, are optional lenses. A model proposal is stored as a proposal. The researcher decides. A field named `judgment` stays empty until a person sets it.

## 1. What already exists

`get_graph("open_code")` is:

`segment_and_shuffle` → `prepare_round` → `Send("code_worker")` → `commit` → `sweep` → `consolidate` → `review_evidence` → `export`.

Workers read one frozen snapshot. `commit` is the only writer. `review_evidence` loops while a code still has unsent assignments and both `max_final_review_calls` (default 8) and `max_final_review_tokens` (default 24000) can take the next prompt. If the allowance stops the loop early, `review_complete` is false, `consolidated` is false, and the result records pending ids. Work already done is kept.

`get_graph("themes")` is a separate reporting graph: `prepare` → `Send("theme_worker")` → `validate`. `build_themes` groups codes already earned. It does not reread passages and it does not mint codes. Theme objects already carry a `properties` list. That list is a reporting field. Later-stage properties go in a different store.

State that later stages must reuse:

| Piece | Current behavior |
| --- | --- |
| Codebook | Stable `code_id`, revisions, definitions, aliases, document ids, assignment lists. |
| Segments | Evidence spans, `quote_status`, located document, provenance of the passage that was being coded. A bad quote is flagged and kept. |
| Memos | `kind` of `passage`, `document`, or provenance. A provenance note records reading order. It is not reflexivity. A sweep may also write a reflexive memo; that memo is the model's text. |
| Comparisons | `alias_duplicate`, `refine_definition`, `split_code`, `link_to_category`, `record_dimension`, `keep_distinct`, `defer`. `relate_categories` is stored and not executed. |
| Saturation | `new_codes_pre_alias` and `new_codes_post_alias` per sweep window. `judgment` is always null. |
| Checkpoints | Manifest of corpus hash, context, model settings, procedure, and `policy_version`. Corpus, `unit`, and `seed` cannot be continued. Other changes need `allow_changes` or `continuation`. |
| Export | `codebook.json`, `codebook.csv`, `taguette_codebook.csv`, `codebook.qdc`, `coded_segments.json`, `highlights.csv`, memos. Taguette imports the codebook CSV. Taguette does not import highlights or `.qdc`. |
| Tests | `tests/fake_llm.py` and `tests/fake_themer.py` return schema-shaped JSON and make no network call. |

`context` may carry a research question and sensitizing concepts. Keys named `codes`, `codebook`, `tags`, or a coding paradigm raise `PresetCodesError`. A later-stage lens is a procedure setting on that later call. It is not a preset codebook, and it is not accepted inside `context` during initial coding.

## 2. What the method literature says

Claims about Charmaz's definitions are taken from the 2006 edition. The 2014 second edition is the edition to cite in a paper. This design does not rest on page-level differences between those two editions. Claims about Glaser's 1978 families and about the 1990 paradigm are taken from authors who describe those books, and those authors are cited. Where a book was not read for this note, the note says so.

### 2.1 Discovery, constant comparison, sampling

Glaser and Strauss (1967) argue for generating theory from data by comparative analysis, with theoretical sampling as the collection of further data directed by the emerging theory, memos, and theoretical saturation as a judgment about a category. Saunders et al. (2018) quote their criterion: saturation means no additional data are being found with which the researcher can develop properties of the category, after the researcher has looked for groups that stretch the diversity of data on that category. The constant comparative method moves through four stages that stay in play together: comparing incidents applicable to each category, integrating categories and their properties, delimiting the theory, and writing the theory (Glaser & Strauss, 1967, Chapter V; read here in the Grounded Theory Review reprint, Glaser & Strauss, 2008). That reprint also says the method is used jointly with theoretical sampling, whether the sampling collects new data or works over qualitative data already in hand. An incident that adds nothing new to a category need not be coded again; that is a judgment about the category, after incidents have been compared. It is not a quota of interviews. When a category is not saturated, the same chapter says the analyst goes back, to the field or to the notes.

The 1967 book also holds two ideas that later split: categories "emerge," and the researcher needs theoretical sensitivity (Kelle, 2005). Kelle's point is that those two ideas were already in tension.

### 2.2 Glaser after 1967

Glaser (1978) *Theoretical Sensitivity* develops incident coding and theoretical sensitivity, and introduces theoretical coding families as ways substantive codes might relate. Charmaz (2006) counts 18 families in that book and quotes the first, the Six Cs: causes, contexts, contingencies, consequences, covariances, and conditions. She quotes Glaser's (1978) definition of theoretical codes as conceptualizing how the substantive codes may relate to each other as hypotheses to be integrated into a theory. Kelle (2007), in the SAGE handbook, adds that the list also includes a strategies family and a means-goal family, and that one term (he uses *goal*) can sit in more than one family. He argues the families are a fund of theoretical ideas, and that their usefulness for relating substantive codes is limited unless the researcher already understands how the terms connect. This design does not transcribe the rest of the 1978 list. An implementation must not invent the missing names. Glaser (2005) is his later book on theoretical coding; Birks and Mills (2015) treat that book as the place Glaser does theoretical coding in the advanced stage. This note does not state a count of families from 2005.

Glaser (1992) *Basics of Grounded Theory Analysis: Emergence vs Forcing* is the book-length charge that Strauss and Corbin force data into a preset frame. Kelle (2005) summarizes that charge and answers it: the charge names real weaknesses, and it overstates the risk. Refusing every prior concept does not solve the problem. Kelle's alternative is abductive inference, statements that have empirical content, and corroboration. Seidel and Urquhart (2013) record Glaser's stage order as open coding, selective coding, then theoretical coding. In that order the core category is chosen before the families are used to relate categories. That is the reverse of Strauss and Corbin's 1990 order.

### 2.3 Strauss and Corbin, then Corbin and Strauss

Strauss and Corbin (1990, 1998) *Basics of Qualitative Research* specify open coding, axial coding, and selective coding. Open coding includes properties and dimensions. Axial coding treats one category as an axis and relates it to subcategories. Kelle (2007) describes their 1990 paradigm model as one general model of action, used to build the axis, with attention to action and interaction strategies and to intentions and goals. Selective coding then relates other categories to a core category. Strauss and Corbin (1990) also use a storyline as a way to integrate the theory; Birks and Mills (2015) keep the storyline for that job.

The author order and the procedure both change. Corbin and Strauss (2008, 2015) are the third and fourth editions. Rieger (2019), comparing the approaches, reports that those editions no longer present open, axial, and selective coding as three required phases; the last phase is called theoretical integration; and the paradigm is a tool rather than a set of directives. The analytic grouping she reports from the 2008 edition is conditions, actions/interactions and emotions, and consequences, plus a conditional/consequential matrix for links between macro and micro conditions. This design treats the paradigm as that kind of tool. It is off unless the caller asks for it. A project that wants the 1990 procedure as its method can set the lens; the package default does not become a 1990 template.

### 2.4 Charmaz

Charmaz (2006) distinguishes initial coding from focused coding. Focused coding uses the most significant and/or frequent earlier codes to sift through large amounts of data. Frequency is one criterion. Analytic significance is the other. Focused coding is also where the researcher checks preconceptions about which initial codes can categorize the data.

She treats axial coding as optional. The intent, bringing fractured codes back together and asking when, where, why, who, how, and with what consequences, can be useful. Applying Strauss and Corbin's frame can also limit what the researcher learns, and it can lay what she calls a technological overlay on the data. She writes that she has not used axial coding as Strauss and Corbin prescribe, and that researchers who can tolerate ambiguity can follow the leads in their analysis without that frame. Theoretical coding, following Glaser, comes after the focused codes and specifies possible relationships. She also warns that an axial coding matrix and any set of Glaser's theoretical codes can force data if they are applied as a prescription (Charmaz, 2006). Codes and families have to earn their place.

Her glossary defines theoretical saturation as the point at which gathering more data about a theoretical category reveals no new properties and yields no further theoretical insights about the emerging theory. In the same chapter she warns that researchers confuse saturation with repetition ("I kept finding the same patterns"), cites Morse (1995) on proclaiming saturation rather than demonstrating it, and reports Dey's (1999) preference for "theoretical sufficiency" because saturation sounds more final than the analysis is. A small study with modest claims is a different saturation claim from a study with a heavy theoretical claim.

Theoretical sampling, for Charmaz, is the move back to the field to develop categories: to fill gaps, extend reach, and check hunches. It starts after categories exist. It is not the initial sample, and it is not a reshuffle of documents chosen in advance.

Memo-writing and sorting are how Charmaz integrates. Memos are the researcher's analytic record. Sorting memos is a later step that arranges the argument. A generated paragraph is not that record.

### 2.5 Handbook and the practical guide

Bryant and Charmaz (2007) collect the competing standpoints, including Glaser on formal theory and Kelle on how categories are developed. Kelle's handbook chapter is the place this design takes the comparison of coding families and the coding paradigm. Birks and Mills (2015) list the methods they treat as essential: initial coding and categorizing, concurrent data collection and analysis, memos, theoretical sampling, constant comparison with inductive and abductive logic, theoretical sensitivity, intermediate coding, identifying a core category, and advanced coding and theoretical integration. They use the storyline (Strauss & Corbin, 1990) and Glaser's (2005) theoretical coding as two ways to do the advanced stage. The 2015 edition is the one whose front matter was checked. A third edition (2023) exists; this note does not take claims from it.

The disagreement that matters for software:

| Question | Glaser | Strauss and Corbin (1990/1998) | Corbin and Strauss (2008/2015), as reported by Rieger (2019) | Charmaz (2006) |
| --- | --- | --- | --- | --- |
| After initial coding | Selective coding delimits to a core, then theoretical coding relates categories with families | Axial coding around a paradigm, then selective coding | Open and axial work is iterative; the closing step is theoretical integration | Focused coding, then theoretical coding; axial coding is optional |
| Prior theory | Families sensitize; a single paradigm forces | One action paradigm is the axis | The paradigm is a tool, not a directive | Prior ideas are acknowledged; frames must earn their way |
| Core category | A basic social process is the usual product | Selective coding integrates around a core; the storyline presents it | Integration, with the storyline still available | Integration by memos, sorting, and diagrams; one core is not a quota |
| Saturation | Properties of the category stop developing | Categories are developed in properties, dimensions, and relations | Same concern, with less stage machinery | No new properties and no new theoretical insights; repetition is not saturation |

Kelle (2005, 2007) is the reason this design does not treat "no paradigm" as the same thing as "no theoretical ideas." The Charmaz default still asks relational questions. It does not fill paradigm slots unless asked.

## 3. Saturation, as the later literature splits it

### 3.1 Four meanings

Saunders et al. (2018) separate four models. Theoretical saturation, the grounded-theory model, is about developing categories, tied to theoretical sampling. Inductive thematic saturation is about new codes or themes appearing. A priori thematic saturation asks whether predetermined themes are exemplified. Data saturation is redundancy in the data, with no necessary reference to a theory. They argue that saturation is a prediction about the unobserved made from the observed, and that the operationalization has to match the study's logic. They also say the idea loses its point if it is stretched across every qualitative design. For this package the matching model is theoretical saturation. The new-code curve already produced at open coding is closer to inductive thematic saturation.

### 3.2 Interview counts are study-specific

Guest, Bunce, and Johnson (2006) operationalized saturation as the point at which no new themes were observed, in a thematic analysis of 60 interviews with women in two West African countries. In that dataset, saturation of themes occurred within the first 12 interviews, and basic elements of metathemes were present by six. The denominator was the themes in that fully analyzed sample. The paper is about nonprobability sample size for interviews. It is not a grounded-theory rule, and 12 is not a constant.

Guest, Namey, and Chen (2020) add a prospective method for inductive thematic analysis: a base size (they tested 4, 5, and 6 interviews), a run length (they tested 2 and 3), and a new-information threshold (they propose ≤5% and 0% as reporting options, not as a law). They point out that a retrospective ratio, of the kind in Guest et al. (2006) and in Hennink et al., reaches 100% by construction once the denominator is every theme in a fully analyzed dataset. On a corpus this package has already coded from end to end, a "percent of codes found" figure will say the codes are saturated because there are no codes left outside the run. That figure is a report of the open-coding curve. It is not theoretical saturation.

### 3.3 Code saturation and meaning saturation

Hennink, Kaiser, and Marconi (2017) distinguish code saturation from meaning saturation in 25 interviews. Code saturation, where the range of thematic issues was identified and the codebook stabilized, was reached at nine interviews. Meaning saturation, where further dimensions, nuances, and insights of those issues stopped appearing, took 16 to 24 interviews. Their phrasing is that code saturation is when researchers have "heard it all," and meaning saturation is when they "understand it all." Saunders et al. (2018) read meaning saturation as nearer to theoretical saturation than code saturation is, and they also say Hennink and colleagues stay at the level of codes rather than theoretical categories, and position the approach outside grounded theory. The numbers 9 and 16–24 belong to that study. This design keeps the distinction and does not import the numbers.

### 3.4 Braun and Clarke

Braun and Clarke (2021) argue that data saturation as information redundancy fits coding-reliability thematic analysis, where meaning is treated as something to excavate and coders apply a codebook to a stopping rule. It does not fit reflexive thematic analysis, where meaning is generated in interpretation and the stopping decision is situated. They include code saturation and meaning saturation in that critique. This package is not reflexive thematic analysis. The critique still binds the software: a curve going flat is evidence a person can read. It is not a decision the software may announce. Charmaz's constructivist stance is closer to "meaning is constructed" than to "meaning is excavated," which is another reason `judgment` stays null.

### 3.5 Low, and Aldiabat and Le Navenec

Low (2019) argues that defining saturation as "no new information" gives the researcher no way to recognize the point, and that it is a fallacy because further data can always yield new theoretical insights. She argues that a large part of the newer literature asks how many interviews are required rather than what saturation is. Fleet and Reeves (2023) quote her rejecting an absolute end point of analysis and tying a pragmatic alternative to conceptual rigor of the theory (Low, 2019, as quoted there, pp. 131, 136). The category log in this design is a way to show whether properties, dimensions, and relations are still moving. It is not a claim that the theory is finished.

Aldiabat and Le Navenec (2018) discuss saturation inside grounded theory and, for the length of their article, use "data saturation" to mean theoretical saturation. They tie reaching it to theoretical sampling and to constant comparison, and they discuss what hinders it. They also restate Hennink and colleagues' interview counts as if they were grounded-theory sample sizes. That restatement is the conflation Saunders et al. (2018) warn against. This design follows Aldiabat and Le Navenec on the dependence on sampling and comparison. It does not follow them on the counts.

Theoretical saturation in this package is a category-level record: new properties, new dimensions, new relations, and negative cases still unexplained, for each category, in each batch. It is not a data-level record of new passages or new code labels.

## 4. Computational grounded theory and LLM coding

### 4.1 Nelson, and Berente, Seidel, and Safadi

Nelson (2020; first published online 2017) proposes computational grounded theory in three steps: computational pattern detection, human pattern refinement by a computationally guided deep reading, and computational pattern confirmation. The interpretation that turns output into a sociological claim sits with the researcher. The middle step is not optional in her account. Dunivin cites the online-first date as 2017; it is the same article.

Berente, Seidel, and Safadi (2019) propose computationally intensive theory development from trace data, iterating sampling, synchronic analysis, lexical framing, and diachronic analysis. Lexical framing is the step in which researchers name what the computational patterns mean. Their material is trace data, not an interview codebook. The useful transfer is the iteration and the human naming step. Their four processes are not axial coding and not a core category.

### 4.2 What the LLM studies actually did

Xiao et al. (2023) used GPT-3 with an expert codebook for deductive coding of children's questions. Agreement with experts was fair to substantial on one task (Cohen's κ = 0.61 for question complexity) and lower on the other (κ = 0.38 for syntactic structure). Codebook-centered prompts beat example-centered prompts. The task assigns a fixed label set. It is not initial or focused coding.

Chew et al. (2023) describe LLM-assisted content analysis for deductive coding. The human codebook comes first. The model applies it. Their paper is a method for supporting that application, including checks that catch codes the model is effectively guessing. It is not a grounded-theory procedure.

De Paoli's verified studies are thematic analysis in Braun and Clarke's sense. The Social Science Computer Review paper appeared online in December 2023 and was published in 2024 (De Paoli, 2024a). GPT-3.5-Turbo was asked to perform an inductive thematic analysis of two already published interview datasets, and it recovered most of the main themes from the earlier human analyses. De Paoli (2024b) follows up in FQS with more open prompts, duplicate-code handling, and thematic maps, again on thematic analysis. This search did not find a De Paoli paper that carries out grounded theory. Those papers are evidence about theme recovery under a thematic-analysis prompt. They are not evidence about properties, dimensions, or a core category.

Dunivin (2024) reports a socio-historical codebook built and reliability-tested by people, then applied by a model. Against the human standard, GPT-4 reached Cohen's κ ≥ 0.79 on 3 of 9 codes and κ ≥ 0.6 on 8 of 9; GPT-3.5 did not (mean κ = 0.34). Asking the model to justify the code (chain-of-thought) improved fidelity. He describes the definition–evaluation–refinement loop as Nelson's pattern. The 2025 journal version of that project (Dunivin, 2025) keeps interpretive leadership with the human researcher and treats the model as a way to apply a codebook at a scale people will not code by hand. Some codes match people. Some do not. A single κ is not a property of "the model."

Tai et al. (2024) start from the same stochastic problem: one completion is a bad summary of a model. They code the same passages across iterations and compare the set with traditional coding. Agreement with people is something to measure over runs. It is not established by one response.

Chen et al. (2024) measure open codes without a ground-truth codebook: merge codebooks, then coverage, overlap, novelty, and divergence. The repository already uses those metrics. They diagnose flooding and codes that do not connect to the data. They measure inductive code sets. They do not measure relations, a core category, or saturation.

Yue, Liu, Lv, Hao, and Cui (2025) walked ChatGPT-4-Turbo through open, axial, and selective coding on interviews with blind gamers, and compared it with manual coding. The abstract reports shortcomings in depth, context, connections, and coding organization. In the tutorial itself, axial coding of eight interviews did not fit in one prompt, so they staged pairwise comparisons and clustering, and they instructed the model to forget an earlier thematic pass and cluster again. Selective coding was a prompt to name a core and link other categories. That is evidence that a chat window will produce axial-looking and selective-looking text, and that connections were a weak point relative to the human coding. It is also evidence that an unconstrained "forget and recluster" prompt is a poor fit for a system that keeps evidence.

### 4.3 Reflexivity and the human in the loop

Nelson (2020), Berente et al. (2019), Dunivin (2024, 2025), and Chew et al. (2023) all keep a person in charge of what the codes mean. Charmaz (2006) locates memo-writing and the saturation claim in the researcher. The current package already separates a provenance note from a reflexive memo, and it will store a reflexive memo when the sweep model writes one.

A model memo about "the analyst's position" is the model's text. Researcher reflexivity is a note the researcher writes about their own position, the sensitizing concepts they brought, and where they revised a code. Later stages store that note under `kind: "researcher_reflexivity"` and store model text under `kind: "model_memo"`. The software does not relabel one as the other.

The same boundary applies to judgment. A prompt that says "decide whether the category is saturated" invites the model to perform the researcher's act. The implementation asks for reasons tied to the category log (which properties moved, which dimension pole has one passage, which relation is single-document). Those reasons land in `proposed_reasons`. `judgment` changes only when the researcher writes it. Calling the model's text a judgment is the anthropomorphism this design refuses. Yue et al. (2025) show the practical version of the same problem: the model will answer a selective-coding prompt with a core category whether or not the links are supported.

Chain-of-thought, as Dunivin (2024) used it, is a rationale attached to a coding decision. Later-stage prompts may require a rationale and citations. The rationale is evidence for review. It is not a raise in confidence, and self-rated confidence stays diagnostic, as it is today.

## 5. Rules that apply to every later stage

1. A later stage is a new graph. It reads a finished open-coding run through `load_run`. It does not insert nodes into `open_code` or `themes`.
2. Workers read a frozen snapshot and return proposals. One commit node writes. The proposal reducer keeps the current rules: dedupe by proposal id, drop conflicting payloads, sort, clear only on a reset.
3. Every analytic claim carries span ids and document ids. Quotes go through the existing anchor. `quote_status` of `possibly_modified` or `unanchored` keeps the claim and flags it. `evidence_outside_passage` keeps the current attribution rules.
4. A claim with weak support is flagged and kept. The open-coding rule is the later-stage rule.
5. Each stage has a call allowance and a token allowance, checked in that order, with the same stop reasons as `review_evidence`: `complete`, `call_allowance`, `token_allowance`, `no_progress`. An exhausted allowance sets the stage `complete` flag false and leaves an incomplete status. Outputs written so far stay. Allowances are ceilings. They are not measured costs. A live corpus has to be priced separately. The focused reread splits its ceiling across three reserved purposes (section 6) and reports coverage for each.
6. The checkpoint manifest gains the stage name, the lens, the allowances, and a hash of any researcher edit applied at the start of the stage. Corpus, `unit`, and `seed` stay non-continuable. A researcher edit is an expected continuation, recorded with old and new values.
7. Tests use a fake model that returns fixed JSON for the new schemas. Tests assert flags, incomplete status, and that `judgment` is still null when the researcher has not set it. They assert that `build_themes` is unchanged and that an open-coding sweep still stores `relate_categories` without executing it.
8. Preset codes stay forbidden on the initial-coding `context`. A lens is a parameter of the later call.

Suggested starting allowances, by analogy with the final-review defaults, not from the method literature: 8 calls and 24000 estimated input tokens for a focused recode; 12 calls and 36000 tokens for axial, because each category carries evidence; 8 calls and 24000 tokens for the selective draft. Zero on either allowance runs the deterministic part and skips the model loop. These numbers are configuration. A live corpus will need them remeasured.

## 6. Focused coding

### What the literature requires

Charmaz (2006) defines focused coding as sifting data with the most significant and/or frequent initial codes, and as a check on which of those codes can actually categorize. Birks and Mills (2015) place this kind of work in intermediate coding, after initial coding and beside constant comparison. Glaser would already be moving toward a core; this stage does not do that. Strauss and Corbin would already be dimensionalizing inside open coding and then axising; property notes may be written here, and relationships wait.

### Best practice

Raise a code because it is frequent, because it spreads across documents, or because it carries analytic weight: a researcher pin, a substantive memo, an in-vivo term, or a contrast the book would lose if only frequent codes survived. Charmaz's "and/or" is the rule. A rare code that holds a negative case stays eligible.

Then reread. The point of the stage is to try the raised codes against passages, including passages that might contradict them, and to revise. The allowance is one ceiling split into three reserved, interleaved purposes: developing existing assignments, testing competing interpretations, and examining passages outside the initial selection. Each queue is sized to its reserved share of the call allowance. Neighbor, outside, and negative-case limits are optional caps, not a default of one. Competing readings are seeded from the initial codings before any property claim exists, and one later round searches claims written by the first reread. A purpose that is still empty, or that uses fewer calls than it reserved, returns those calls explicitly, and the result warns. Coverage reports the calls actually used. Passages queued and not sent are `focused_not_reread`. Passages never queued are `focused_not_queued`. `complete` is corpus coverage. `complete_for_queue` means the queue finished and says so. Researcher-selected incidents, from the interrupt resume or from `focused_incidents.csv`, can enter any of the three purposes. `seed` orders the queue. When the candidates fit, it does not change membership. `focused_sampling_seed` samples only when a list is longer than its slot.

### Pitfalls

Selecting on frequency alone buries the variation saturation is supposed to stretch (Glaser & Strauss, 1967, as quoted by Saunders et al., 2018). Raising codes by a model "importance" score, with no pin and no memo, repeats the anthropomorphism problem. Recoding that silently replaces the initial code destroys the audit of what changed. Treating the result as a theme report collapses this stage into `build_themes`.

### Implementation

`get_graph("focused")`:

`load_open_coding` → `select_candidates` → `researcher_checkpoint` → `prepare_recode` → `Send("recode_worker")` → `commit_focused` → `review_recode` → `claim_checkpoint` → `export_focused`.

`researcher_checkpoint` and `claim_checkpoint` call `interrupt()` inside the node. Resume is `Command(resume=...)`. An empty object is not a resume value, because LangGraph treats a dict whose keys are interrupt ids, and an empty dict matches.

`select_candidates` is deterministic. Inputs, all already on the codebook or the memo list:

- assignment frequency
- share of documents and of respondents
- count of substantive memos on the code. This count can reward a verbose memo. `focused_analytic_features` can drop `memo`
- in-vivo flag, dropped by ablating `in_vivo`
- stored comparisons (`refine_definition`, `record_dimension`, `link_to_category`, `keep_distinct`), dropped by ablating `comparison`
- researcher pin, written into the run before the graph starts
- a contrast flag when the code is the sole counter-example on a dimension already recorded

Weights are config (`focused_weight_frequency`, `focused_weight_documents`, `focused_weight_analytic`) and the function is documented beside the code. The analytic term is the mean of the features that remain. Pins are included even when the score is low. The default score rule is `top_k`: the upper half, ties broken by `code_id`, and not every code. `focused_selection_rule=quantile` keeps that ceiling. `focused_raise_all` is what raises every code. An explicit `focused_min_score` is a hard `>=` cutoff. A selection that raises at least `focused_raise_warn_fraction` (default 0.75) warns. The output is `raised_code_ids` plus `not_raised_code_ids`. Nothing is deleted from the codebook.

`prepare_recode` builds three queues and interleaves them by share (`focused_share_develop`, `focused_share_compete`, `focused_share_outside`). `focused_share_claim_compete` is reserved for the round that runs after properties exist and is not interleaved into that first queue. The four shares sum to 1. The default is one quarter each. A three-purpose mapping that already sums to 1 keeps claim compete at 0. The same seed replays the order.

- `develop`: passages already assigned to a raised code, except those moved into `compete`, up to that purpose's reserved share
- `compete`: passages that test a competing interpretation, up to that purpose's reserved share. Before claims exist, the seeds are a different valence or consequence of the same code, a low-confidence or flagged assignment, another code already on the passage, and an opposite pole. Within-category negative cases and lexical neighbors fill the same share. `focused_negative_case_limit` and `focused_neighbor_limit` cap those sources when set. `None` does not cap them. The compete prompt lists the codebook on equal footing, alternatives first, and does not mark raised codes. Each passage requires a keep, revise, or drop verdict with a rationale for every initial code, and a required strongest alternative. Revise and drop are flagged and not applied. A kept raised code is flagged `alternative_named_not_chosen` when the memo or that alternative names a different code
- `claim_compete`: after the first reread writes properties, the reserved share searches those claims against other passages. Its coverage is reported with the other purposes. It does not run by consuming leftover calls from the first three shares
- `outside`: passages outside the initial selection, up to that purpose's reserved share, capped by `focused_outside_limit` when that cap is set. Hits from a claim's negative-case search fill this quota before other unassigned passages. When the cap is positive and the list is empty, the lowest-confidence assignments of raised codes fill it

Negative-case search is per claim. It searches passages inside the claim's supporting category, including the same code with an opposite consequence, and passages outside that support. It does not search only outside a storyline, and it does not search only passages that were already flagged. Researcher incidents from the resume value or from `focused_incidents.csv` are placed in the purpose the researcher names and are not dropped to meet a limit. `seed` replays order. When the candidate list fits the reserved slots, that seed does not change membership. `focused_sampling_seed` chooses which rows are kept only when the list is longer than the slot. The queue stops when the next prompt would break the allowance. Queued passages that were not sent are `focused_not_reread`. Passages never queued are `focused_not_queued`. `purpose_coverage` reports queued, reread, not reread, calls, tokens, the requested share, and the applied share (calls used divided by calls used in total). A purpose that received fewer calls than it reserved is `reallocated_away`, with a warning.

`recode_worker` sees the frozen book and the passage. Develop and outside see raised codes and the others and are told to prefer an existing code. Compete and claim compete list the codebook on equal footing. A compete passage returns one verdict per initial code and a strongest alternative. `revise` and `drop` are flagged and are not applied without review. The prompt asks for analytic properties and dimensions with a document id, passage id, and quote, and for a competing interpretation. It does not ask for a descriptive summary. `commit_focused` appends assignments. Re-applying a code on the same span records a confirmation on the existing assignment and does not add a second row. A different span is nested on the original assignment. A narrowed quote that drops words of the original is flagged `narrowed_span`. Downstream assignment rows do not grow a parallel segment for that proposal. A new code with lexical similarity at or above `focused_duplicate_similarity` (default 0.5) to an existing code is flagged `possible_duplicate`. An initial assignment that the recode does not repeat is kept and flagged `focused_disagreement`. The researcher resolves it later. New labels follow today's validation: retry once, then keep and flag. They do not bypass `PresetCodesError` rules; the raised list is a lens on an earned book, and a brand-new label is still earned from the passage. Loading the open-coding folder restores segment and memo id counters, and new memos use `m00001`. The export writes segments, assignments, and the prompt log with the raw model response, plus `focused_claims.csv` and `downstream_claims.json`. A model call uses `focused_request_timeout` as one wall-clock budget for streaming and retries, and `focused_request_retries` as the retry cap inside that budget.

Property and dimension notes are a log. Properties that share a category and a `property_key` merge and keep each citation. `property_stability` reports category-level and name-level Jaccard across runs; a single run records the keys and leaves the scores empty. Each row keeps span ids, a `category_revision_id`, and an `evidence_revision_id`. `focused_tokens_used` is provider-reported input, cached, and output. `focused_tokens_estimated` is the ceiling checked before the call. Property notes are not relations, and they are not theme `properties`. The store is `category_development.json`, `state_version` 1. `relations` is present and null. A later stage may fill it. Integration reads this state and does not require `load_axial`.

Each proposal row keeps three separate fields. `disposition` is `proposed`, `accepted`, `rejected`, or `superseded`. `evidence_condition` is `anchored`, `disputed`, or `insufficient`. `downstream_eligible` is true only when the disposition is `accepted`, the evidence condition is `anchored`, and the row is not `needs_review`. Rows are kept. `category_development.json` and `focused_claims.csv` use every row. `downstream_claims.json` uses only eligible rows. Each file states which records it uses.

`claim_support` is a researcher label: `supported`, `partially_supported`, `contradicted`, or `indeterminate`. Until that review it is `undefined`, and reports say undefined. The model does not set it. Citation anchoring rate is the share of quotes that locate a span (`exact`, `normalized_match`, or `fuzzy_match`). Tentative and unanchored quotes are counted beside it. Citation anchoring is not claim support. A property whose `category_id` is not among the codes assigned to the cited passage is flagged `property_category_mismatch` and `needs_review`. The quote can still be anchored. `category_match` is reported beside the anchoring rate so the two are not read as the same number.

A researcher edit to a category appends the previous label and definition to `revision_history`, bumps `category_revision_id`, and marks dependent claims and assignments `needs_review`. It does not erase the claim text or the quote. `record_revision_cycle` records a return to evidence the same way. A new run that adds data is `plan_linked_run`: it cites the parent directory and the sampling brief and does not rewrite the parent.

This graph is distinct from `build_themes` because it rereads passages and can add assignments. `build_themes` remains the reporting view over whatever book it is given.

## 7. Axial coding: properties, dimensions, relations

### What the literature requires

Strauss and Corbin (1990, 1998) make this the stage for properties, dimensions, subcategories, and the links among them, with the paradigm as the axis in those editions. Charmaz (2006) will ask the same relational questions without the slots, and she will skip the stage when the analysis can proceed without it. Glaser (1978, 1992) treats a required paradigm as forcing, and would use a family only as a theoretical code after a core is in view (Seidel & Urquhart, 2013). Kelle (2005, 2007) says both the paradigm and the families are theoretical scaffolds, and that the risk is applying them as a mold rather than as a hypothesis with empirical content.

Rieger's (2019) account of Corbin and Strauss (2008, 2015) is the version that matches an optional lens: the paradigm is a tool; conditions, actions/interactions, and consequences are available; the conditional/consequential matrix is for researchers who want macro–micro links, and it is heavy.

### Best practice

For each raised category, write properties and dimensions that name a range, and subcategories, each tied to passages. Then write relationships as hypotheses. A relationship evidenced in one document can matter; Glaser and Strauss treat a single modifying incident as analytically real. The software flags it `tentative_few_documents` when fewer than `min_relation_documents` distinct documents (default 2) contain anchored evidence. The relationship stays. Two documents is a review threshold for this package. It is not a rule from the method literature.

### Pitfalls

Filling every paradigm slot produces a complete-looking diagram with empty cells, which is the forcing Glaser (1992) and Charmaz (2006) describe. Yue et al. (2025) show a practical version: axial coding became clustering and theme trees under a token limit, and connections were the weak point. Executing the open-coding op `relate_categories` would hide that history; those rows were stored without a theory being built. A model statement of a causal link without two cited passages is an unsupported inference; the existing flag `unsupported_inference` already names that case.

### Implementation

`get_graph("axial")`, only after a focused run:

`load_focused` → `prepare_category` → `Send("category_worker")` → `commit_development` → `prepare_relations` → `Send("relation_worker")` → `commit_relations` → `review_relations` → `export_axial`.

`coding_lens` defaults to `charmaz`. The category prompt asks what varies, under what conditions the category changes, and what it is not. It does not mention paradigm slot names.

`coding_lens="paradigm"` adds optional fields `condition`, `action_interaction`, and `consequence`, matching the 2008 grouping Rieger (2019) reports. `paradigm_version="1990"` adds the finer labels `causal_condition`, `context`, `intervening_condition`, and `strategy` for a project that has chosen the 1990 book as its method. Empty slots stay empty and are flagged `paradigm_slot_empty`. They are not filled to complete the diagram. `coding_lens="glaser_families"` may attach one family name. The shipped names are only those this note can source: the Six Cs (Charmaz, 2006) and strategies and means-goal (Kelle, 2007). Any other family is `family_name` free text supplied by the researcher, or proposed by the model and stored as `family_status: "proposed"`. The commit does not apply a proposed family until the researcher sets `family_status: "accepted"`.

Each property, dimension pole, subcategory, and relation is a record: stable id, category ids, text, span ids, document ids, `quote_status`, and flags. Dimension poles name the ends of a range and cite the passages that sit at those ends. A relation records `from_id`, `to_id`, `relation_type` (a short label such as "varies with" or, under the paradigm lens, a slot name), and the evidence. `min_relation_documents` (default 2) sets `tentative_few_documents` when the anchored document count is lower. The row remains.

Relation prompts are bounded the way sweep chunks are bounded. One request carries the two categories, their definitions, and a capped set of evidence (`axial_evidence_limit`, default the same idea as `cross_chunk_limit`). Further pairs wait for another request. A pair is not decided from one side's evidence.

Fifty categories make 1,225 unordered pairs. The allowance cannot examine that set. `plan_relation_pairs` takes categories in salience order. Adjacent categories are the obvious band. The other pairs are the less-obvious band. Each round mixes both bands (`obvious_share`, default 0.5). Pairs not chosen are reported as unexamined, with ids, not only as a count. Examined pairs are not repeated. The call and token allowances that stop the rounds are ceilings, not measured costs. A round that would exceed the ceiling is not sent. The unexamined list stays on the result.

Relation rows use the same proposal fields as properties: disposition, evidence condition, and downstream eligibility. Core centrality, when selective coding scores candidates, is reported twice: once over proposed relations and once over researcher-accepted relations. The two numbers are not collapsed. Relation stability, in the evaluation table, compares direction, relation type, and conditions. It does not compare endpoint pairs alone. Two runs that name the same categories in opposite directions, or under different conditions, are not the same relation.

`relate_categories` rows already on the open-coding sweep log are copied into the axial input as hints. `commit_relations` does not treat them as finished relations.

Theme `properties` are not read or written here.

## 8. Selective coding and the core category

### What the literature requires

Strauss and Corbin (1990) integrate around a core and use a storyline to say how the categories fit. Birks and Mills (2015) keep both the core and the storyline among the essential and advanced methods. Glaser's order puts selective coding earlier, to delimit the core, and uses theoretical coding afterward (Seidel & Urquhart, 2013). Charmaz (2006) integrates by sorting memos and by theoretical codes that have earned their place. One core category is a common product in the Glaser and Strauss traditions. It is not a success condition for a Charmaz-informed run. A study can return two candidates and leave the choice open.

Yue et al. (2025) asked the model to perform selective coding from axial results and to highlight a dominant category. Their own comparison found connections weak. A storyline the model writes in one pass is a draft.

### Best practice

Score candidates so the researcher can see centrality, reach, and explanatory variation. Centrality is how many other categories have a relation to this one, reported separately for proposed relations and for relations the researcher has accepted. Reach is document and respondent coverage of its evidence. Explanatory variation is how many properties and dimension poles the category organizes, including poles that contradict a simple story. The scores are a table. The core is whatever the researcher writes into `researcher_core_id`. The field ships empty. Several candidates can stay in play.

The storyline is a memo with links to category ids and span ids. An adversarial pass searches each claim, including passages inside its supporting category that show an opposite consequence, and proposes those passages as negative cases. The researcher keeps, revises, or sets aside the storyline. Analytic induction and constant comparison both take negative cases seriously; Glaser and Strauss distinguished their method from analytic induction, and the pass here is a search for misfits, not a claim to have proved the storyline by eliminating every alternative.

### Pitfalls

A weighted sum that picks the winner forces a core the same way a paradigm forces a process. Premature closure is the saturation failure Charmaz (2006) and Low (2019) describe: a label is declared central before its properties have been developed. A model "attack" on the storyline is not a negative case. A negative case is a passage. The model may nominate passages; the passage is the evidence.

### Implementation

`get_graph("selective")` reads a versioned category-development state. Relations on that state are optional. The graph does not require a node named `load_axial`. A run that never wrote relations still loads. `state_version` is checked. A missing version is read as 1.

`load_category_development` → `score_candidates` → `draft_storyline` → `negative_case` → `export_selective`.

Returning to evidence is a recorded cycle, `return_to_evidence`, `revise_claim`, or `researcher_edit`. The cycle names the claim ids and passage ids. Dependent rows gain `needs_review`. Their text stays. A sampling brief that needs data the corpus does not contain is a new linked run (`plan_linked_run`). The new run stores `parent_run_id` and does not rewrite the parent directory.

`score_candidates` is deterministic from the category-development records. Centrality is counted twice, once for proposed relations and once for relations whose disposition is `accepted`. The two scores stay in separate fields. It writes `core_candidates` with the scores and the ids behind them. It does not set a core.

`draft_storyline` is one bounded model call, or several if the candidate set does not fit, under the selective allowance. The memo `kind` is `model_memo`, `role` is `storyline_draft`, and every sentence that names a category carries ids. Missing citations flag `storyline_uncited`. The export banner says the storyline is a draft for revision.

`negative_case` searches each relationship and property claim, not the storyline as a single blob. For each claim it retrieves passages inside that claim's supporting category, including the same code with an opposite consequence or another pole of the same dimension, and passages outside that support. Passages already flagged `focused_disagreement` or `tentative_few_documents` are included. They are not the whole search. The model proposes which of those passages strain the draft. Each proposal stores the span ids and the claim id. The researcher disposition (`keep_storyline`, `revise`, `set_aside`) is a later edit, default empty.

Glaserian order, core before families, is a configuration note in section 14. It is not a second graph in the first three implementation PRs.

## 9. Saturation

### What the literature requires

The record is category-level, as Glaser and Strauss (1967) and Charmaz (2006) define it: properties, and theoretical insight, not a new label count. Saunders et al. (2018) require the report to say which model of saturation is being used. Hennink et al. (2017) require code saturation and meaning saturation to be reported apart, with Saunders's warning that meaning saturation is still not theoretical saturation. Guest et al. (2020) require a prospective rule if a threshold is used, and they show why a retrospective percentage on a finished corpus is automatic. Braun and Clarke (2021) and Low (2019) require a person to make the stopping claim, and they deny that "no new information" is that claim. Aldiabat and Le Navenec (2018) require the claim to stay tied to comparison and to sampling, not to a sample-size formula.

### Best practice

Keep the open-coding curve (`new_codes_pre_alias`, `new_codes_post_alias`) and label it code saturation, in Hennink's sense, which Saunders would class with inductive thematic saturation. Add a category log, per batch of later-stage work: new properties, new dimension poles, new relations, and negative cases still unexplained. Label that log as the material of a theoretical-saturation decision. A Guest-style base, run, and threshold may be computed on the open-coding curve only, during open coding, and labeled as a thematic-saturation diagnostic. After the corpus is fully coded, do not report a percentage of codes found against the full book as if it were a stopping rule.

`proposed_reasons` lists the categories still moving and the categories whose last several batches added no property, dimension, or relation. `judgment` stays null. The researcher sets it, with their own sentence, in `researcher_saturation`.

### Pitfalls

Declaring saturation because new codes hit zero is the mistake the current `saturation.py` docstring already refuses. Importing 12, or 9, or 16–24, as a default, repeats a result from one study. Letting the model write `judgment: true` is the anthropomorphism section 4.3 refuses. Stopping a fixed corpus early, because a code curve flattened, conflicts with the package rule that every document is read, and it confuses data saturation with theoretical saturation.

### Implementation

Extend `saturation_point` and `saturation_report`. Existing keys stay. New keys:

- `new_properties`, `new_dimensions`, `new_relations`, `unexplained_negative_cases`, each per batch and per category
- `code_saturation`: the label-count curve, plus an optional Guest report `{base_size, run_length, new_information_threshold, ratio, below_threshold}` computed only on open-coding windows
- `category_saturation`: the property and relation log
- `proposed_reasons`: list of strings tied to category ids
- `judgment`: null until `record_saturation_decision` writes `researcher_saturation`

`record_saturation_decision(run_dir, decision, reasons)` is the only writer of `judgment`. Allowed decisions are `not_claimed`, `sufficient_for_this_question`, and `not_saturated`. There is no `true`. The wording follows Charmaz's point that a modest claim and a heavy claim are different, and Low's point that there is no absolute end.

Incomplete later stages leave `category_saturation.status` as `incomplete` and name the allowance that stopped them. A partial log is still a log.

## 10. Theoretical sampling inside a fixed corpus

### What the literature requires

Theoretical sampling directs further analysis by the categories already earned (Glaser & Strauss, 1967; Charmaz, 2006). Aldiabat and Le Navenec (2018) treat it as part of how saturation is reached. The constant comparative chapter allows that sampling on new data or on qualitative data already collected (Glaser & Strauss, 2008). When a category is thin, the analyst can return to the notes or return to the field. Charmaz (2006) stresses the return to the field: sampling fills gaps that the initial sample was not built to fill, and it starts after categories exist. A seed shuffle of a corpus chosen in advance is neither of those returns. The methodology document already says the default run is not theoretical sampling.

Inside a corpus that will not grow, both returns have a software shape. The return to the notes is a second queue aimed at thin categories. The return to the field is a brief that names the questions and the sites the corpus cannot supply. The queue is theoretical sampling of data already in hand. The brief is the part of theoretical sampling this package cannot carry out.

### Best practice

Rank passages for that second pass by the gap they might fill for a named thin category: a property with one pole, a relation that is `tentative_few_documents`, a negative case the storyline does not cover. The score is expected information for that category. A generic novelty rank over the whole corpus would recreate data saturation (Saunders et al., 2018).

Do not drop passages from the open-coding read. The second queue does not replace it. If the allowance ends first, the status is `incomplete` and the ids not read in this pass are listed.

The sampling brief names, for each thin category, what is missing and a question or a site that could supply it. It does not say the data were collected.

### Pitfalls

Reordering the original coding queue on resume is already forbidden: corpus, `unit`, and `seed` fix the passages and the order stored at `segment_and_shuffle`. The pursuit queue is a new list on the later-stage checkpoint. It does not rewrite that order. Prioritizing passages by embedding similarity, with no thin category named, is retrieval. It is not a return to the notes for a category. A brief that recommends interviews without saying which category is thin is a generic methods paragraph.

### Implementation

`pursuit` is a node sequence on the selective graph, or a small graph of its own if selective coding is not being run: `rank_gaps` → `queue_pursuit` → optional `Send("pursuit_worker")` → `write_brief`.

`rank_gaps` reads `category_development.json` and emits gaps `{category_id, gap_type, evidence_ids}`. `queue_pursuit` ranks passages not yet used as evidence for that gap, using the existing retriever, and stores the queue. `pursuit_worker` is a reread with the same flag-and-keep rules as focused recoding. It can be skipped (`pursuit="brief_only"`, the default) so the first implementation writes the brief without another coding bill.

`sampling_brief.md` and `sampling_brief.json` list gaps, the passages already tried, and suggested questions. `kind` of the memo is `model_memo` when the model drafts a question, and the researcher can replace the question in the JSON. The brief's status line says new data were not collected.

## 11. Human checkpoints and the edit round-trip

Charmaz's memos, Nelson's pattern refinement, and Berente and colleagues' lexical framing are checkpoints between machine passes. LangGraph already supports `interrupt_before`. The later graphs interrupt before `commit_focused`, before `commit_relations`, and before `export_selective`. The researcher resumes with an unchanged manifest or with a recorded edit. A model call cannot clear an interrupt.

The round-trip uses the files Taguette can actually import, plus a stable-id sidecar this package writes and Taguette does not read. `taguette_codebook.csv` carries labels and descriptions. `code_id_sidecar.json` maps each tag to `code_id` and `category_revision_id`. A researcher may rename, rewrite a definition, pin a code, or mark a code `not_raised`, then reimport. The import hashes the file into the manifest, stores the previous text, and flags `researcher_edited`. A code missing from the imported file is flagged `researcher_removed` and kept, with its segments, until an approved mapping lists that `code_id` under `removals`. A new tag and a missing tag are not a rename. They are a removal plus an addition until `renames` in the approved mapping names the `code_id` and the new tag. `classify_codebook_roundtrip` is that check. Silent deletion would break the flag-and-keep rule. A category edit from that mapping marks dependent claims and assignments `needs_review` and keeps `revision_history`.

`.qdc` stays an export for other QDA tools. Taguette does not import it, and this package still does not write a Taguette SQLite project. `highlights.csv` remains an export of evidence. Editing a highlight and expecting it to re-enter the run is out of scope until an import for that file exists. The open question is whether that import is worth building, or whether researchers should edit `category_development.json` directly.

Researcher reflexivity is an optional text file `researcher_reflexivity.md` read at the start of a stage and stored as `kind: "researcher_reflexivity"`. If it is absent, the run records that it is absent. The software does not draft a substitute.

## 12. Evaluation

Metrics answer different questions. None of them is a saturation decision.

| Check | What it measures | What it does not measure |
| --- | --- | --- |
| Citation anchoring rate | Share of properties, dimensions, relations, and storyline claims whose spans anchor as `exact`, `normalized_match`, or `fuzzy_match`. Tentative and unanchored claims are reported beside it, not folded in. | Whether the claim is true. That question is claim support. |
| Claim support | Researcher review of each claim: `supported`, `partially_supported`, `contradicted`, or `indeterminate`. Unreviewed claims are `undefined` and are reported as undefined. | A model score. The model does not write this field. |
| Researcher ratings | A small sheet, `researcher_ratings.json`: forcing, usefulness, and whether the claim was recognizable in the cited passages. Stored as the researcher's numbers. Correction time is recorded by the researcher and stays undefined until then. | A model score of quality. |
| Cross-seed stability | Jaccard of `raised_code_ids`. For relations, stability compares direction, relation type, and conditions. Endpoint pairs alone are not the comparison. Instability is reported. | Validity. A constructivist reading expects some variation (Charmaz, 2006). |
| Chen et al. (2024) coverage, overlap, novelty, divergence | Open codebooks, as now. | Relations, cores, or saturation. |
| Agreement on a fixed grid | Reuse `aligned_assignment_agreement` when a second coder, human or model, codes the same passages with the raised set. | A license to skip the researcher. Tai et al. (2024) are the reason a live comparison uses more than one model completion. |

A comparison with a published grounded-theory study needs a public codebook, public excerpts or spans, and a stated school (Glaser, Strauss and Corbin, or Charmaz). This design does not name a dataset that meets that bar. The Muse figures in `design.md` are an open-coding comparison from the 0.5.4 era. They are not a benchmark for properties, relations, or a core. Using them as one would over-claim.

The offline test of the implementation is the fake model: known passages, known citations, a relation with one document flagged tentative, `judgment` still null, an exhausted allowance leaving `incomplete`, and `build_themes` still refusing to reread.

`scripts/evaluate_focused.py` and [focused-evaluation.md](focused-evaluation.md) compare initial coding alone with a focused reread on a small corpus: category revisions, citation-anchored properties, claim support (undefined until reviewed), counterexamples, system disagreements kept in the published example, researcher correction time (undefined until recorded), token cost, and cost per supported claim. Analytic-weight ablation drops memo count, the in-vivo flag, or stored comparisons. Memo count can reward verbosity. The script estimates a live run and does not call an API. A person approves a live run before one is started. The estimate for `grok-4.20-0309-non-reasoning` is about $1.98, under a $2 ceiling. That figure is a plan, not a bill. Allowances remain ceilings.

## 13. Proposed API and config

```python
focused = await afocused_code(
    open_coding_result,
    model=model,
    seed=7,
    coding_lens="charmaz",  # "paradigm" | "glaser_families"
    max_focused_review_calls=8,
    max_focused_review_tokens=24000,
    focused_share_develop=0.25,
    focused_share_compete=0.25,
    focused_share_outside=0.25,
    focused_share_claim_compete=0.25,
    focused_outside_limit=1,
    focused_negative_case_limit=1,
    focused_analytic_features=("memo", "in_vivo", "comparison"),
    interrupt=True,
)
axial = await aaxial_code(
    focused,
    model=model,
    coding_lens="charmaz",
    paradigm_version="2008",  # used only when lens is "paradigm"; "1990" is the other value
    min_relation_documents=2,  # flag threshold
    axial_evidence_limit=8,
    max_axial_review_calls=12,
    max_axial_review_tokens=36000,
    interrupt=True,
)
selective = await aselective_code(
    axial,
    model=model,
    max_core_candidates=5,
    pursuit="brief_only",  # or "reread"
    max_selective_review_calls=8,
    max_selective_review_tokens=24000,
    interrupt=True,
)
record_saturation_decision(selective.output_dir, "not_claimed", "Properties of paying-attention are still moving.")
```

Synchronous wrappers follow the existing `open_code` / `aopen_code` pattern. `get_graph` accepts `"focused"`, `"axial"`, and `"selective"`.

Manifest keys added when those calls exist: `stage`, `coding_lens`, `paradigm_version`, `min_relation_documents`, the three allowance pairs, `pursuit`, and `researcher_edit_hash`. They ride in the procedure block the way `max_final_review_calls` does.

`researcher_core_id` and `researcher_saturation` are absent until written. Export adds `category_development.json`, `relations.json`, `core_candidates.json`, `storyline.md`, `negative_cases.json`, `sampling_brief.md`, and an updated saturation object on the result. `taguette_codebook.csv` gains a column only if Taguette's importer will ignore unknown columns; that has to be checked against the pinned Taguette version before the column is added. Until then, pins and "not raised" live in `focused_selection.csv`, which this package imports, and the Taguette file stays the label-and-description file it is today.

## 14. Phased implementation

The phases follow Charmaz's order: focused recoding, then relations, then integration and the saturation claim. A Glaserian project would select a core before applying families. That order can be a later configuration. It is not part of these three PRs, because building both orders at once doubles the graphs before either has tests.

**PR1. Focused coding and the property log.** `afocused_code`, candidate selection, allowance-bounded recode, `focused_disagreement` and `focused_not_reread`, property and dimension rows with citations in `category_development.json`. No relations and no core. Fake-model tests. `build_themes` untouched.

**PR2. Axial relations.** `aaxial_code`, the three lenses, bounded relation requests, `tentative_few_documents`, paradigm slots only when requested, Glaser family names limited to the sourced list plus researcher free text. Open-coding `relate_categories` stays stored-only.

**PR3. Core, saturation, pursuit.** `aselective_code`, candidate scores, storyline draft, negative-case pass, category-level saturation log, `record_saturation_decision`, within-corpus pursuit queue, sampling brief. `judgment` remains null unless that function is called.

Each PR carries its manifest keys and its incomplete-status tests before any live run is interpreted.

## 15. Where this design changes the brief

The brief asked for a literature-grounded design of those five stages on the current architecture. Checking the sources changed several of the implementation choices inside that brief.

- A relationship with evidence from one document is flagged `tentative_few_documents` and kept. Two documents is a review threshold, not a validity gate. One modifying incident is analytically real in Glaser and Strauss's comparative method.
- Core scores do not select a core. `researcher_core_id` ships empty. More than one candidate can remain. Charmaz does not set a quota of one core; Strauss and Corbin's selective coding does, and that choice stays with the person.
- A second pass over a finished corpus is theoretical sampling of data already in hand, which the constant comparative chapter allows (Glaser & Strauss, 2008). It is aimed at named thin categories. The sampling brief is the return to the field: data the corpus does not contain (Charmaz, 2006). A generic information-gain rank over the whole corpus is not that sampling.
- Pursuit does not drop passages and does not rewrite the open-coding order fixed by corpus, `unit`, and `seed`.
- Focused-code selection uses frequency, document spread, and analytic weight. A rare contrast code stays eligible (Charmaz, 2006, "significant and/or frequent").
- Property rows are a new store. Theme `properties` stay a reporting field.
- `relate_categories` on the open-coding sweep is not promoted into a theory.
- `judgment` stays null until `record_saturation_decision`. The model writes `proposed_reasons`.
- Guest et al. (2006, 2020) and Hennink et al. (2017) are reported as code-saturation and meaning-saturation diagnostics. Their interview counts are not defaults. A retrospective percentage on a fully coded corpus is not a stopping rule (Guest et al., 2020).
- Meaning saturation is not recorded as theoretical saturation (Saunders et al., 2018).
- De Paoli (2024a, 2024b) is thematic analysis. This note does not cite those papers as grounded-theory results.
- A model memo is not researcher reflexivity.
- The focused reread is bounded by an allowance, and passages it skips are flagged `focused_not_reread`.
- The paradigm defaults off, which matches Charmaz (2006) and the later Corbin and Strauss position as Rieger (2019) reports it. It does not match a project that has adopted Strauss and Corbin (1990) as its method. That project sets `coding_lens="paradigm"` and `paradigm_version="1990"`.
- Glaser family names shipped in code are only the Six Cs and the strategies and means-goal families, plus free text. The rest of the 1978 list is not guessed.
- Cross-seed instability is a report. It is not a failed run.
- No published grounded-theory dataset is named as the evaluation corpus. The bar is a public codebook, public excerpts, and a stated school.

## 16. Open questions

1. Is the Charmaz order (focused, then relations, then a core) the order to implement, or should a Glaserian order (core, then families) be a second graph in the same series of PRs?
2. The focused reread now reserves shares for developing assignments, competing interpretations, and passages outside the selection, and it reports coverage per purpose. Should a purpose be required to finish its own queue before its unused calls move, even when that returns `incomplete` for the other purposes?
3. Should researcher edits re-enter only through `focused_selection.csv` and the Taguette codebook CSV, or is a `highlights.csv` import in scope?
4. May a run keep multiple cores without a warning, or should an empty `researcher_core_id` at export raise a warning the researcher can silence?
5. Is `min_relation_documents=2` the right flag threshold for the first axial PR?
6. Should `record_saturation_decision` be a Python call, a JSON file dropped into the run directory, or both?
7. Is there a published grounded-theory study, with a public codebook and public excerpts, that should be the later-stage evaluation corpus?
8. Should the sampling brief be allowed to call the model, or should PR3 write it deterministically from the gap list so the brief does not add a prose pass?
9. The starting allowances (8/24000, 12/36000, 8/24000) are ceilings by analogy with final review. They are not measured costs. The live evaluation plan in [focused-evaluation.md](focused-evaluation.md) prices one small comparison near $2 before anyone approves the calls. What corpus should replace that plan?

## 17. References

Aldiabat, K. M., & Le Navenec, C.-L. (2018). Data saturation: The mysterious step in grounded theory method. *The Qualitative Report, 23*(1), 245–261. https://doi.org/10.46743/2160-3715/2018.2994

Berente, N., Seidel, S., & Safadi, H. (2019). Research commentary—Data-driven computationally intensive theory development. *Information Systems Research, 30*(1), 50–64. https://doi.org/10.1287/isre.2018.0774

Birks, M., & Mills, J. (2015). *Grounded theory: A practical guide* (2nd ed.). SAGE. (A 3rd edition was published in 2023. Claims in this note are from the 2015 front matter.)

Braun, V., & Clarke, V. (2021). To saturate or not to saturate? Questioning data saturation as a useful concept for thematic analysis and sample-size rationales. *Qualitative Research in Sport, Exercise and Health, 13*(2), 201–216. https://doi.org/10.1080/2159676X.2019.1704846 (Published online 26 December 2019.)

Bryant, A., & Charmaz, K. (Eds.). (2007). *The SAGE handbook of grounded theory*. SAGE.

Charmaz, K. (2006). *Constructing grounded theory: A practical guide through qualitative analysis*. SAGE.

Charmaz, K. (2014). *Constructing grounded theory* (2nd ed.). SAGE.

Chen, J., Lotsos, A., Cheng, S., Wang, C., Zhao, L., Hullman, J., Sherin, B., Wilensky, U., & Horn, M. (2024). *A computational method for measuring "open codes" in qualitative analysis* (arXiv:2411.12142). https://doi.org/10.48550/arXiv.2411.12142

Chew, R., Bollenbacher, J., Wenger, M., Speer, J., & Kim, A. (2023). *LLM-assisted content analysis: Using large language models to support deductive coding* (arXiv:2306.14924). https://arxiv.org/abs/2306.14924

Corbin, J., & Strauss, A. (2008). *Basics of qualitative research: Techniques and procedures for developing grounded theory* (3rd ed.). SAGE.

Corbin, J., & Strauss, A. (2015). *Basics of qualitative research: Techniques and procedures for developing grounded theory* (4th ed.). SAGE.

De Paoli, S. (2024a). Performing an inductive thematic analysis of semi-structured interviews with a large language model: An exploration and provocation on the limits of the approach. *Social Science Computer Review, 42*(4), 997–1019. https://doi.org/10.1177/08944393231220483 (Published online 7 December 2023.)

De Paoli, S. (2024b). Further explorations on the use of large language models for thematic analysis. Open-ended prompts, better terminologies and thematic maps. *Forum Qualitative Sozialforschung / Forum: Qualitative Social Research, 25*(3). https://doi.org/10.17169/fqs-25.3.4196

Dey, I. (1999). *Grounding grounded theory: Guidelines for qualitative inquiry*. Academic Press. (Cited from Charmaz, 2006, on theoretical sufficiency. Not re-read for this note.)

Dunivin, Z. O. (2024). *Scalable qualitative coding with LLMs: Chain-of-thought reasoning matches human performance in some hermeneutic tasks* (arXiv:2401.15170). https://arxiv.org/abs/2401.15170

Dunivin, Z. O. (2025). Scaling hermeneutics: A guide to qualitative coding with LLMs for reflexive content analysis. *EPJ Data Science, 14*, Article 28. https://doi.org/10.1140/epjds/s13688-025-00548-8

Fleet, D., & Reeves, A. (2023). Adapting Strauss and Corbin's grounded theory in clinical case study research in the field of counselling. *Counselling and Psychotherapy Research, 23*(4), 874–881. https://doi.org/10.1002/capr.12642

Glaser, B. G. (1978). *Theoretical sensitivity: Advances in the methodology of grounded theory*. Sociology Press.

Glaser, B. G. (1992). *Basics of grounded theory analysis: Emergence vs forcing*. Sociology Press.

Glaser, B. G. (2005). *The grounded theory perspective III: Theoretical coding*. Sociology Press. (Cited from Birks & Mills, 2015, as the advanced-coding treatment. Family counts from this book are not stated here.)

Glaser, B. G., & Strauss, A. L. (1967). *The discovery of grounded theory: Strategies for qualitative research*. Aldine. Routledge reprint: https://doi.org/10.4324/9780203793206

Glaser, B. G., & Strauss, A. L. (2008). The constant comparative method of qualitative analysis. *Grounded Theory Review, 7*(3). https://groundedtheoryreview.org/index.php/gtr/article/view/467 (Reprint of Glaser & Strauss, 1967, Chapter V, first published in *Social Problems, 12*, 1965, pp. 436–445. The four stages and the remark on sampling already collected data are taken from this reprint.)

Guest, G., Bunce, A., & Johnson, L. (2006). How many interviews are enough? An experiment with data saturation and variability. *Field Methods, 18*(1), 59–82. https://doi.org/10.1177/1525822X05279903

Guest, G., Namey, E., & Chen, M. (2020). A simple method to assess and report thematic saturation in qualitative research. *PLOS ONE, 15*(5), Article e0232076. https://doi.org/10.1371/journal.pone.0232076

Hennink, M. M., Kaiser, B. N., & Marconi, V. C. (2017). Code saturation versus meaning saturation: How many interviews are enough? *Qualitative Health Research, 27*(4), 591–608. https://doi.org/10.1177/1049732316665344

Kelle, U. (2005). "Emergence" vs. "forcing" of empirical data? A crucial problem of "Grounded Theory" reconsidered. *Forum Qualitative Sozialforschung / Forum: Qualitative Social Research, 6*(2), Article 27. https://doi.org/10.17169/fqs-6.2.467

Kelle, U. (2007). The development of categories: Different approaches in grounded theory. In A. Bryant & K. Charmaz (Eds.), *The SAGE handbook of grounded theory* (pp. 191–213). SAGE.

Low, J. (2019). A pragmatic definition of the concept of theoretical saturation. *Sociological Focus, 52*(2), 131–139. https://doi.org/10.1080/00380237.2018.1544514

Morse, J. M. (1995). The significance of saturation. *Qualitative Health Research, 5*(2), 147–149. https://doi.org/10.1177/104973239500500201 (Cited from Charmaz, 2006, and from Saunders et al., 2018. Not re-read for this note.)

Nelson, L. K. (2020). Computational grounded theory: A methodological framework. *Sociological Methods & Research, 49*(1), 3–42. https://doi.org/10.1177/0049124117729703 (First published online 2017.)

Rieger, K. L. (2019). Discriminating among grounded theory approaches. *Nursing Inquiry, 26*(1), Article e12261. https://doi.org/10.1111/nin.12261

Saunders, B., Sim, J., Kingstone, T., Baker, S., Waterfield, J., Bartlam, B., Burroughs, H., & Jinks, C. (2018). Saturation in qualitative research: Exploring its conceptualization and operationalization. *Quality & Quantity, 52*(4), 1893–1907. https://doi.org/10.1007/s11135-017-0574-8

Seidel, S., & Urquhart, C. (2013). On emergence and forcing in information systems grounded theory studies: The case of Strauss and Corbin. *Journal of Information Technology, 28*(3), 237–260. https://doi.org/10.1057/jit.2013.17

Strauss, A., & Corbin, J. (1990). *Basics of qualitative research: Grounded theory procedures and techniques*. SAGE.

Strauss, A., & Corbin, J. (1998). *Basics of qualitative research: Techniques and procedures for developing grounded theory* (2nd ed.). SAGE.

Tai, R. H., Bentley, L. R., Xia, X., Sitt, J. M., Fankhauser, S. C., Chicas-Mosier, A. M., & Monteith, B. G. (2024). An examination of the use of large language models to aid analysis of textual data. *International Journal of Qualitative Methods, 23*. https://doi.org/10.1177/16094069241231168

Xiao, Z., Yuan, X., Liao, Q. V., Abdelghani, R., & Oudeyer, P.-Y. (2023). Supporting qualitative analysis with large language models: Combining codebook with GPT-3 for deductive coding. In *Proceedings of the 28th International Conference on Intelligent User Interfaces* (pp. 75–78). ACM. https://arxiv.org/abs/2304.10548

Yue, Y., Liu, D., Lv, Y., Hao, J., & Cui, P. (2025). A practical guide and assessment on using ChatGPT to conduct grounded theory: Tutorial. *Journal of Medical Internet Research, 27*, Article e70122. https://doi.org/10.2196/70122
