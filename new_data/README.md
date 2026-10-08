# Role-swap dataset for agent/patient binding

A controlled English dataset for studying how language models bind **thematic roles** (agent, patient) to **fillers** (the participants of an event). It is built for:

- estimating agent/patient centroids or role directions in model activations,
- steering those representations to flip the model's answer (e.g. *"Mary kicked John. Who performs the action?"* → *John*),
- role–filler / tensor product representation (TPR) analyses, which need a small, fixed filler vocabulary.

Every item is a **quad**: one event in four versions, with the same participants in both active and passive voice, and with their roles swapped.

| condition | example | agent | patient |
|---|---|---|---|
| `active` | Mary kicked John. | Mary | John |
| `passive` | John was kicked by Mary. | Mary | John |
| `active_swap` | John kicked Mary. | John | Mary |
| `passive_swap` | Mary was kicked by John. | John | Mary |

Because the active and passive versions come in pairs, **thematic role** and **linear position / syntactic subject** vary independently. The swapped versions use exactly the same words, so the only thing that changes is who is bound to which role.

---

## Files

| file | contents |
|---|---|
| `make_role_swap_dataset.py` | Generator script. Deterministic, standard library only (Python ≥ 3.8). |
| `role_swap_long.csv` | **One row per sentence** (7,668 rows by default), with all annotations. Use this one for analyses. |
| `role_swap_quads.csv` | **One row per quad** (1,917 rows by default), with the four sentences side by side. Useful for inspection and paired analyses. |

### Regenerating

```bash
python make_role_swap_dataset.py                      # default: past tense, all subsets
python make_role_swap_dataset.py --tense present      # "Mary kicks John." / "John is kicked by Mary."
python make_role_swap_dataset.py --tense both         # both tenses (doubles the size)
python make_role_swap_dataset.py --no-anomalous       # sensible subsets only (5,808 sentences)
python make_role_swap_dataset.py --mixed-names 0      # drop the mixed name–noun subset
python make_role_swap_dataset.py --include-noun-agentive   # + occupation nouns with agentive verbs
python make_role_swap_dataset.py --out-prefix my_set  # -> my_set_long.csv, my_set_quads.csv
```

The script prints a JSON summary of counts. It **fails with an assertion error** if any of these checks fails:
- no duplicate sentences within a subset;
- every character span matches its NP;
- within each subset, every filler appears equally often as agent and patient, and as NP1 and NP2, in each voice;
- in the mixed subset, name-as-agent and noun-as-agent occur equally often;
- every sensible sentence violates no selectional restriction, and every anomalous sentence violates at least one (see `anomaly_locus`);
- no anomalous sentence also appears among the sensible ones.

To change the vocabulary, edit the lists at the top of the script (`NAMES`, `ANIMATE_NOUNS_CONTACT`, `INANIMATE_NOUNS`, `AGENTIVE_VERBS`, `CONTACT_VERBS`, `SUBSTANCE_VERBS`) and the selectional-restriction sets (`AGENT_MUST_BE_ANIMATE`, `PATIENT_MUST_BE_ANIMATE`, `PATIENT_MUST_BE_SUBSTANCE`), which drive both the anomalous subsets and the `anomaly_locus` annotation.

---

## Subsets

Default build: past tense, 22 fillers, 23 verbs, 48 word types. All words are real English words.

| subset | fillers | verbs | animacy (agent→patient) | semantic status | sentences |
|---|---|---|---|---|---|
| `names_agentive` | 10 names | 12 agentive | AA | sensible | 2,160 |
| `nouns_contact` | 6 animate + 6 inanimate nouns | 8 contact | AA, AI, IA, II | sensible | 2,112 |
| `mixed_contact` | 4 names × 12 nouns | 8 contact | AA, AI, IA | sensible | 1,536 |
| `anomalous_II` | 6 inanimate nouns | 7 agentive (animate agent required) | II | anomalous | 420 |
| `anomalous_AI_IA` | 6 animate × 6 inanimate nouns | 5 agentive (animate agent *and* patient required) | AI, IA | anomalous | 720 |
| `anomalous_AA` | 10 names | 4 substance verbs | AA | anomalous | 720 |
| **total** | | | | | **7,668** |

Examples of the anomalous subsets (each in all four versions):

| subset | example | swapped | `anomaly_locus` |
|---|---|---|---|
| `anomalous_II` | The rock thanked the door. | The door thanked the rock. | both (agent for *kick*, *call*) |
| `anomalous_AI_IA` | The ball thanked the farmer. | The farmer thanked the ball. | agent / patient |
| `anomalous_AA` | David peeled Mary. | Mary peeled David. | patient |

### Vocabulary

- **Names:** Mary, John, Anna, David, Sarah, Peter, Emma, James, Lisa, Tom. `mixed_contact` uses the first four (2 female, 2 male).
- **Animate nouns** (contact subsets): teacher, doctor, farmer, pilot, chef, driver.
- **Inanimate nouns:** ball, rock, box, car, door, branch.
- **Agentive verbs:** kick, push, pull, tickle, chase, follow, call, thank, praise, blame, warn, help.
- **Contact verbs:** hit, strike, touch, block, push, bump, scratch, brush.
- **Substance verbs** (the only words added for the anomalous subsets): sip, peel, fold, melt. Their patient must be a substance or object, which none of the fillers is. *drink* is avoided because *was drunk by* is ambiguous with the adjective.
- **Anomalous verb sets** (taken from the agentive verbs):
  - `anomalous_II`: kick, tickle, call, thank, praise, blame, warn. All require an animate agent.
  - `anomalous_AI_IA`: tickle, thank, praise, blame, warn. These require an animate agent and an animate patient.

The anomalous subsets add **no new fillers**. They reuse the fillers of the sensible subsets, so role/filler analyses share one filler inventory.

### Design rationale

- **Reversible events.** Every event is equally plausible in both orders. All verbs are non-symmetric (no *meet*, *marry*), and none is tied to an occupation (no *examine*, *teach*). This means world knowledge can't tell the model who did what.
- **Agentive verbs** have a clear doer and undergoer, so "who performs the action?" has an unambiguous answer. They only take animate agents, so they're used with names.
- **Contact/causation verbs** accept both animate and inanimate agents (*The branch scratched the pilot*). This is what makes all four animacy configurations possible.
- **Mixed name–noun pairs** test whether role representations transfer across NP forms (a bare name vs *the* + noun). Names and nouns are agent and patient equally often, so NP form is uncorrelated with role.
- **Anomalous subsets** are grammatical, use only real words, and make no sense, because a verb's selectional restrictions are violated. In every anomalous quad, **both orders are anomalous**, so the swap does not give the model a plausible alternative to fall back on. Role can be assigned by syntax (word order, *was …ed by*) but not from world knowledge. Each subset covers a different animacy configuration:
  - `anomalous_II`: an inanimate agent of a verb that needs an animate one (*The rock thanked the door*).
  - `anomalous_AI_IA`: an inanimate participant in either role of a verb that needs animate participants. *The ball thanked the farmer* has the violation on the agent; *The farmer thanked the ball* has it on the patient. This lets you test whether steering depends on where the anomaly sits.
  - `anomalous_AA`: two animate participants with a verb that needs a substance as patient (*David peeled Mary*). Animacy here looks like a normal transitive event, so only the verb–patient fit is broken.

---

## Columns of `role_swap_long.csv`

| column | description |
|---|---|
| `quad_id` | Shared by the 4 versions of one event. |
| `condition` | `active`, `passive`, `active_swap`, `passive_swap`. |
| `subset` | See [Subsets](#subsets). |
| `semantic_status` | `sensible` or `anomalous`. |
| `sensible` | 1 if `semantic_status == sensible`, else 0. |
| `anomaly_locus` | Which selectional restriction is violated: `none`, `agent`, `patient`, or `both`. Computed per sentence, so it changes between the base and swapped versions in `anomalous_AI_IA`. |
| `verb_class` | `agentive`, `contact`, or `substance`. |
| `np_type` | `name`, `noun`, or `mixed` (subset-level NP type). |
| `tense` | `past` or `present`. |
| `voice` | `active` or `passive`. |
| `swapped` | 0 for the base version, 1 for the role-swapped version (relative to `filler_A` = agent). |
| `verb` | Verb lemma. |
| `sentence` | The sentence. |
| `agent`, `patient` | Fillers bound to each thematic role (bare form, without *the*). |
| `agent_animacy`, `patient_animacy` | `animate` or `inanimate`. |
| `animacy_config` | Agent + patient animacy: `AA`, `AI`, `IA`, `II`. |
| `animacy_hierarchy` | `aligned` (animate agent, inanimate patient), `inverted` (inanimate agent, animate patient), or `equal`. |
| `np1`, `np2` | Fillers in linear order. In actives NP1 = agent; in passives NP1 = patient (surface subject) and NP2 is inside the *by*-phrase. |
| `np1_role`, `np2_role` | `agent` or `patient`. |
| `np1_animacy`, `np2_animacy` | Animacy by position. |
| `agent_np_form`, `patient_np_form`, `np1_form`, `np2_form` | `name` or `the_noun`. |
| `np1_char_span`, `np2_char_span`, `verb_char_span` | Character offsets `start:end` of the filler word (without *the*) and the inflected verb (main verb/participle) in `sentence`. Use them to locate token positions with your tokenizer's offset mapping. |
| `logical_form` | e.g. `kick(agent=Mary, patient=John)`. |
| `q_agent`, `q_patient` | Abstract questions: *Who performs / receives the action?* (*Who or what …* for subsets that include inanimates). |
| `q_agent_verb`, `q_patient_verb` | Verb-specific questions: *Who kicked? / Who was kicked?*. These are always phrased in the same form regardless of the sentence's voice. |
| `q_agent_answer`, `q_patient_answer` | Gold answers. They apply to both question types. |

### Columns of `role_swap_quads.csv`

`quad_id`, `subset`, `semantic_status`, `verb_class`, `tense`, `verb`, `filler_A`, `filler_B`, `animacy_A`, `animacy_B`, `active_sentence`, `passive_sentence`, `active_swap_sentence`, `passive_swap_sentence`. In the base versions `filler_A` is the agent; in the swapped versions it is the patient.

---

## Usage notes

### Balance and confounds

- **Role is balanced against everything else.** Within each subset, every filler is agent and patient equally often, in NP1 and NP2, in both voices. Pooled over the sensible data, AI and IA have equal counts, and NP form is balanced across roles.
- **Verb class is tied to subset.** Among the sensible subsets, agentive verbs only appear with names. For animacy comparisons on sensible data, filter to `verb_class == "contact"`.
- **In `mixed_contact` AI/IA items, animacy equals NP form.** The animate participant is always a name and the inanimate one is always *the* + noun. Use `nouns_contact` to estimate pure animacy effects, and the mixed subset for transfer tests.
- **Cell sizes are unequal** (180–1,080 sentences per subset × animacy config × voice), and `names_agentive` is the largest. When computing centroids, average within cells (subset × animacy config × voice) and then across cells. Paired differences within a quad are better still, because they cancel filler identity.
- `push` appears in both verb classes. Filter on `verb_class`, not on the verb lemma.

### Suggested analyses

1. **Role vs position.** Fit the role direction on actives, then test or steer on passives. If it only works within voice, it is tracking subject/position, not thematic role.
2. **Role vs animacy.** Estimate role directions within AA and within II (`nouns_contact`), and compare them with the animacy direction (cosine similarity, cross-steering). Steering AI→IA works against the animacy prior; AA↔AA does not.
3. **Transfer across NP forms.** Apply directions from `names_agentive` to `mixed_contact` and `nouns_contact`.
4. **Sense vs syntax.** Fit directions on sensible items and test them on anomalous ones. If a direction still separates roles and still steers in anomalous sentences, it tracks syntactic role assignment rather than event plausibility. Within `anomalous_AI_IA`, compare items by `anomaly_locus`.
5. **Consistency of the flip.** After steering, both the agent and the patient questions, in both the abstract and the verb-specific form, should flip.

### Before running experiments

- **Tokenization.** Check that the fillers are single tokens (with a leading space) in your model's tokenizer. If one isn't, use the last subtoken or mean-pool over the span.
- **Answer format.** Gold answers for nouns are the bare noun (`teacher`), but models usually answer *"The teacher"*. Score the noun token after *the*, or accept both forms. A fixed prompt ending such as `Answer:` makes the answer start at the first generated token.
- **Plausibility filter.** Some items carry residual priors (social verbs like *praise/blame*; IA items like *The ball blocked the farmer*). Compare each sentence's log-probability with its swapped version and flag quads with large asymmetries.
- **Baseline accuracy.** Report accuracy before steering per subset × voice × question type, and only steer items the model gets right. If accuracy drops on passives, the model is using a "first NP = agent" heuristic. On the anomalous subsets, low accuracy means the model is not assigning roles there, so treat those items as a diagnostic rather than steering targets.
- **Name gender.** Name pairs mix same-gender and different-gender combinations. Check for gender effects and stratify if needed.
- **Held-out splits.** For generalization claims, hold out some verbs and fillers when estimating directions and evaluate on the rest.

---

## Related work

The design draws on:

- Smolensky (1990), tensor product representations
- McCoy et al. (2019), TPDNs ([arXiv:1812.08718](https://arxiv.org/abs/1812.08718))
- Soulos et al. (2020), ROLE ([arXiv:1910.09113](https://arxiv.org/abs/1910.09113))
- Feng & Steinhardt (2024), binding IDs (ICLR 2024)
- Feng et al. (2024), propositional probes ([arXiv:2406.19501](https://arxiv.org/abs/2406.19501))
- Dai et al. (2024, 2026), binding subspaces and cell-based binding (EMNLP 2024; ACL 2026)
- Gur-Arieh et al. (2026), positional/lexical/reflexive retrieval mechanisms (ICLR 2026)
- Geiger et al. (2024), Distributed Alignment Search ([arXiv:2303.02536](https://arxiv.org/abs/2303.02536))
- Hewitt & Manning (2019), structural probes
- Diego-Simón et al., polar probes ([arXiv:2605.14125](https://arxiv.org/abs/2605.14125))
- Acevedo et al. (2026), syntactic/semantic centroids ([arXiv:2601.04765](https://arxiv.org/abs/2601.04765))
