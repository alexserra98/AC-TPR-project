"""
Generate a fully crossed agent/patient dataset for role-filler (TPR) analyses,
with animacy and semantic-anomaly annotation.

Each "quad" = one event type (verb + unordered pair of fillers {A, B}) in 4 versions:
    active          A verb B.            agent=A patient=B
    passive         B was verbed by A.   agent=A patient=B
    active_swap     B verb A.            agent=B patient=A
    passive_swap    A was verbed by B.   agent=B patient=A

Sensible subsets
----------------
names_agentive : names x agentive verbs (kick, thank, praise, ...). AA.
nouns_contact  : 6 animate + 6 inanimate nouns x contact verbs (hit, push, block, ...),
                 which accept animate or inanimate agents. AA, AI, IA, II.
mixed_contact  : names x (animate + inanimate nouns) x contact verbs.
nouns_agentive : (optional, --include-noun-agentive) animate nouns x agentive verbs.

Anomalous subsets (real words, grammatical, semantically nonsensical)
---------------------------------------------------------------------
No new fillers; only selectional restrictions of the verbs are violated.
anomalous_II    : inanimate x inanimate nouns x verbs needing an animate agent
                  ("The rock thanked the door"). Anomalous in both orders.
anomalous_AI_IA : animate x inanimate nouns x verbs needing an animate agent AND an
                  animate patient ("The rock praised the teacher" / "The teacher
                  praised the rock"). Anomalous in both orders, at different loci.
anomalous_AA    : names x verbs needing a substance/object patient
                  ("Mary peeled John"). Anomalous in both orders.

Within each subset every filler appears equally often as agent/patient,
as NP1/NP2, and in active/passive, with every verb.
"""
import argparse
import csv
import itertools
import json
from collections import Counter

NAMES = ["Mary", "John", "Anna", "David", "Sarah",
         "Peter", "Emma", "James", "Lisa", "Tom"]
ANIMATE_NOUNS_AGENTIVE = ["teacher", "doctor", "lawyer", "farmer", "pilot",
                          "chef", "singer", "painter", "driver", "banker"]
ANIMATE_NOUNS_CONTACT = ["teacher", "doctor", "farmer", "pilot", "chef", "driver"]
INANIMATE_NOUNS = ["ball", "rock", "box", "car", "door", "branch"]

# lemma: (3sg present, past, past participle)
AGENTIVE_VERBS = {
    "kick":   ("kicks",   "kicked",   "kicked"),
    "push":   ("pushes",  "pushed",   "pushed"),
    "pull":   ("pulls",   "pulled",   "pulled"),
    "tickle": ("tickles", "tickled",  "tickled"),
    "chase":  ("chases",  "chased",   "chased"),
    "follow": ("follows", "followed", "followed"),
    "call":   ("calls",   "called",   "called"),
    "thank":  ("thanks",  "thanked",  "thanked"),
    "praise": ("praises", "praised",  "praised"),
    "blame":  ("blames",  "blamed",   "blamed"),
    "warn":   ("warns",   "warned",   "warned"),
    "help":   ("helps",   "helped",   "helped"),
}
CONTACT_VERBS = {
    "hit":     ("hits",      "hit",       "hit"),
    "strike":  ("strikes",   "struck",    "struck"),
    "touch":   ("touches",   "touched",   "touched"),
    "block":   ("blocks",    "blocked",   "blocked"),
    "push":    ("pushes",    "pushed",    "pushed"),
    "bump":    ("bumps",     "bumped",    "bumped"),
    "scratch": ("scratches", "scratched", "scratched"),
    "brush":   ("brushes",   "brushed",   "brushed"),
}
# The only new words: verbs whose patient must be a substance/object.
# ("drink" avoided: passive "was drunk by" is ambiguous with the adjective.)
SUBSTANCE_VERBS = {
    "sip":  ("sips",  "sipped", "sipped"),
    "peel": ("peels", "peeled", "peeled"),
    "fold": ("folds", "folded", "folded"),
    "melt": ("melts", "melted", "melted"),
}

# Selectional restrictions (used both to build anomalous subsets and to annotate
# every sentence, so sensible subsets are verified to have no violations).
AGENT_MUST_BE_ANIMATE = {"kick", "tickle", "call", "thank", "praise", "blame", "warn",
                         "chase", "follow", "help", "pull", "sip", "peel", "fold", "melt"}
PATIENT_MUST_BE_ANIMATE = {"tickle", "thank", "praise", "blame", "warn"}
PATIENT_MUST_BE_SUBSTANCE = set(SUBSTANCE_VERBS)   # none of our fillers qualifies

II_VERBS = ["kick", "tickle", "call", "thank", "praise", "blame", "warn"]
AI_IA_VERBS = ["tickle", "thank", "praise", "blame", "warn"]

ANIMACY = {w: "animate" for w in NAMES + ANIMATE_NOUNS_AGENTIVE + ANIMATE_NOUNS_CONTACT}
ANIMACY.update({w: "inanimate" for w in INANIMATE_NOUNS})

QUESTIONS = {
    "who":         ("Who performs the action?", "Who receives the action?"),
    "who_or_what": ("Who or what performs the action?", "Who or what receives the action?"),
}


def np(filler, sentence_initial):
    if filler in NAMES:
        return filler
    return ("The " if sentence_initial else "the ") + filler


def build(np1, np2, forms, voice, tense):
    pres, past, part = forms
    if voice == "active":
        v = pres if tense == "present" else past
        return f"{np(np1, True)} {v} {np(np2, False)}.", v
    aux = "is" if tense == "present" else "was"
    return f"{np(np1, True)} {aux} {part} by {np(np2, False)}.", part


def span(sentence, word, start=0):
    i = sentence.index(word, start)
    return i, i + len(word)


def short(anim):
    return "A" if anim == "animate" else "I"


def form(filler):
    return "name" if filler in NAMES else "the_noun"


def hierarchy(agent_anim, patient_anim):
    if agent_anim == patient_anim:
        return "equal"
    return "aligned" if agent_anim == "animate" else "inverted"


def anomaly(verb, agent, patient):
    ag = verb in AGENT_MUST_BE_ANIMATE and ANIMACY[agent] == "inanimate"
    pa = ((verb in PATIENT_MUST_BE_ANIMATE and ANIMACY[patient] == "inanimate")
          or verb in PATIENT_MUST_BE_SUBSTANCE)
    return {(False, False): "none", (True, False): "agent",
            (False, True): "patient", (True, True): "both"}[(ag, pa)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tense", choices=["past", "present", "both"], default="past")
    ap.add_argument("--include-noun-agentive", action="store_true")
    ap.add_argument("--mixed-names", type=int, default=4,
                    help="how many names (from the start of NAMES) to pair with nouns "
                         "in the mixed subset; 0 disables it")
    ap.add_argument("--no-anomalous", action="store_true",
                    help="drop the semantically anomalous subsets")
    ap.add_argument("--out-prefix", default="role_swap")
    args = ap.parse_args()

    tenses = ["past", "present"] if args.tense == "both" else [args.tense]
    contact_nouns = ANIMATE_NOUNS_CONTACT + INANIMATE_NOUNS

    def pick(d, keys):
        return {k: d[k] for k in keys}

    subsets = [
        # name, filler pairs, verbs, verb_class, question set, semantic status
        ("names_agentive", list(itertools.combinations(NAMES, 2)),
         AGENTIVE_VERBS, "agentive", "who", "sensible"),
        ("nouns_contact", list(itertools.combinations(contact_nouns, 2)),
         CONTACT_VERBS, "contact", "who_or_what", "sensible"),
    ]
    if args.mixed_names > 0:
        subsets.append(("mixed_contact",
                        list(itertools.product(NAMES[:args.mixed_names], contact_nouns)),
                        CONTACT_VERBS, "contact", "who_or_what", "sensible"))
    if args.include_noun_agentive:
        subsets.append(("nouns_agentive",
                        list(itertools.combinations(ANIMATE_NOUNS_AGENTIVE, 2)),
                        AGENTIVE_VERBS, "agentive", "who", "sensible"))
    if not args.no_anomalous:
        subsets += [
            ("anomalous_II", list(itertools.combinations(INANIMATE_NOUNS, 2)),
             pick(AGENTIVE_VERBS, II_VERBS), "agentive", "who_or_what", "anomalous"),
            ("anomalous_AI_IA", list(itertools.product(ANIMATE_NOUNS_CONTACT, INANIMATE_NOUNS)),
             pick(AGENTIVE_VERBS, AI_IA_VERBS), "agentive", "who_or_what", "anomalous"),
            ("anomalous_AA", list(itertools.combinations(NAMES, 2)),
             SUBSTANCE_VERBS, "substance", "who", "anomalous"),
        ]

    long_rows, wide_rows = [], []
    quad_id = 0
    for subset, pairs, verbs, verb_class, qset, semantic in subsets:
        q_ag, q_pat = QUESTIONS[qset]
        wh = "Who" if qset == "who" else "Who or what"
        np_types = {form(x) for p in pairs for x in p}
        np_type = "mixed" if len(np_types) > 1 else ("name" if "name" in np_types else "noun")
        for tense in tenses:
            for verb, forms in verbs.items():
                for a, b in pairs:
                    versions = {
                        "active":       (a, b, "active", 0),
                        "passive":      (a, b, "passive", 0),
                        "active_swap":  (b, a, "active", 1),
                        "passive_swap": (b, a, "passive", 1),
                    }
                    wide = {"quad_id": quad_id, "subset": subset, "semantic_status": semantic,
                            "verb_class": verb_class, "tense": tense, "verb": verb,
                            "filler_A": a, "filler_B": b,
                            "animacy_A": ANIMACY[a], "animacy_B": ANIMACY[b]}
                    for cond, (agent, patient, voice, swapped) in versions.items():
                        np1, np2 = (agent, patient) if voice == "active" else (patient, agent)
                        s, verb_form = build(np1, np2, forms, voice, tense)
                        n1s, n1e = span(s, np1)
                        n2s, n2e = span(s, np2, n1e)
                        vs, ve = span(s, verb_form, n1e)
                        ag_an, pa_an = ANIMACY[agent], ANIMACY[patient]
                        long_rows.append({
                            "quad_id": quad_id,
                            "condition": cond,
                            "subset": subset,
                            "semantic_status": semantic,
                            "sensible": int(semantic == "sensible"),
                            "anomaly_locus": anomaly(verb, agent, patient),
                            "verb_class": verb_class,
                            "np_type": np_type,
                            "tense": tense,
                            "voice": voice,
                            "swapped": swapped,
                            "verb": verb,
                            "sentence": s,
                            "agent": agent,
                            "patient": patient,
                            "agent_animacy": ag_an,
                            "patient_animacy": pa_an,
                            "animacy_config": short(ag_an) + short(pa_an),
                            "animacy_hierarchy": hierarchy(ag_an, pa_an),
                            "np1": np1, "np2": np2,
                            "np1_role": "agent" if np1 == agent else "patient",
                            "np2_role": "agent" if np2 == agent else "patient",
                            "np1_animacy": ANIMACY[np1],
                            "np2_animacy": ANIMACY[np2],
                            "agent_np_form": form(agent),
                            "patient_np_form": form(patient),
                            "np1_form": form(np1),
                            "np2_form": form(np2),
                            "np1_char_span": f"{n1s}:{n1e}",
                            "np2_char_span": f"{n2s}:{n2e}",
                            "verb_char_span": f"{vs}:{ve}",
                            "logical_form": f"{verb}(agent={agent}, patient={patient})",
                            "q_agent": q_ag,
                            "q_agent_answer": agent,
                            "q_patient": q_pat,
                            "q_patient_answer": patient,
                            "q_agent_verb": f"{wh} {forms[0] if tense == 'present' else forms[1]}?",
                            "q_patient_verb": f"{wh} {'is' if tense == 'present' else 'was'} {forms[2]}?",
                        })
                        wide[f"{cond}_sentence"] = s
                    wide_rows.append(wide)
                    quad_id += 1

    with open(f"{args.out_prefix}_long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(long_rows)
    with open(f"{args.out_prefix}_quads.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wide_rows[0].keys()))
        w.writeheader()
        w.writerows(wide_rows)

    # ---- sanity checks ----
    sents = [(r["subset"], r["sentence"]) for r in long_rows]
    assert len(sents) == len(set(sents)), "duplicate sentences within a subset"
    for r in long_rows:
        for k in ("np1", "np2"):
            a, b = map(int, r[f"{k}_char_span"].split(":"))
            assert r["sentence"][a:b] == r[k]
        # sensible items violate no restriction; anomalous items violate at least one
        assert (r["anomaly_locus"] == "none") == bool(r["sensible"]), r["sentence"]
    sensible_sents = {r["sentence"] for r in long_rows if r["sensible"]}
    assert not any(r["sentence"] in sensible_sents for r in long_rows if not r["sensible"])
    for subset in {r["subset"] for r in long_rows}:
        rows = [r for r in long_rows if r["subset"] == subset]
        cells = Counter()
        for r in rows:
            cells[(r["agent"], "agent", r["voice"])] += 1
            cells[(r["patient"], "patient", r["voice"])] += 1
            cells[(r["np1"], "np1", r["voice"])] += 1
            cells[(r["np2"], "np2", r["voice"])] += 1
        for filler in {k[0] for k in cells}:
            vals = {v for k, v in cells.items() if k[0] == filler}
            assert len(vals) == 1, f"{subset}: {filler} unbalanced {vals}"
        forms_by_role = Counter((r["agent_np_form"], r["patient_np_form"]) for r in rows)
        assert forms_by_role[("name", "the_noun")] == forms_by_role[("the_noun", "name")]

    vocab = set()
    for _, s in sents:
        vocab.update(w.strip(".").lower() for w in s.split())
    summary = {
        "quads": len(wide_rows),
        "sentences": len(long_rows),
        "fillers": len({r["agent"] for r in long_rows}),
        "verbs": len({r["verb"] for r in long_rows}),
        "vocab_size": len(vocab),
        "by_subset": dict(Counter(r["subset"] for r in long_rows)),
        "by_semantic_status": dict(Counter(r["semantic_status"] for r in long_rows)),
        "by_anomaly_locus": dict(Counter(r["anomaly_locus"] for r in long_rows)),
        "by_subset_x_config": {f"{k[0]}:{k[1]}": v for k, v in sorted(
            Counter((r["subset"], r["animacy_config"]) for r in long_rows).items())},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
