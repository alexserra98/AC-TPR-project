# Acevedo et al 2026 and the role steering experiment

This note summarizes the centroid method and its connection to the planned
agent/patient intervention. Source: Santiago Acevedo, Alessandro Laio, and Marco
Baroni, [Differential syntactic and semantic encoding in LLMs, arXiv v5](https://arxiv.org/pdf/2601.04765v5).

## Paper method and findings

At each layer, the authors form a syntactic centroid by averaging sentence
representations with a shared part-of-speech template. Semantic centroids average
translations of the same sentence. Originals are excluded from their centroids;
semantic centroids also exclude the English paraphrase.

Sentence representations concatenate or average token states. Ablation removes
the projection onto a centroid:

\[
x_{\perp c} = x - \frac{x^\top c}{c^\top c}c.
\]

The authors measure changes in neighborhood similarity between matched
sentences. Syntax signals extend across more layers; semantics is strongest
centrally. Removing semantic centroids leaves syntax relatively intact, whereas
syntax removal affects semantics more. Main results use DeepSeek V3; appendices
include smaller models and Pythia 6.9B training checkpoints. Appendix D discusses
spurious similarity signals from direct centroid subtraction. These are
representation analyses, with no demonstrated agent/patient steering result.

## Adaptation for this project

The proposed experiment estimates token-level means over examples of each role:

\[
\mu^\ell_r = \frac{1}{N_r}\sum_{i:\operatorname{role}(i)=r}h_\ell(i),
\qquad r\in\{\mathrm{agent},\mathrm{patient}\}.
\]

It then applies the proposed agent-to-patient replacement
`h - mu_agent + mu_patient`, or its reverse, and measures the counterfactual
answer logit minus the original answer logit. This translation differs from
the paper's projection ablation; its effect must be established experimentally.

Agent and patient are semantic roles. Active and passive are grammatical voices.
For “The boy helped the girl.” and “The girl was helped by the boy.”, the agent
remains `boy` and the patient remains `girl`. Whether these role means capture
useful syntax is a hypothesis.

The generated corpus uses every noun in both roles and both voices. Keeping
reversed-role groups together preserves this balance in each split. Four nouns
are reserved for generalization evaluation. Centroid estimation uses training
examples only; held-out examples remain outside the means.

Balance does not eliminate context effects: in a causal language model, a noun
at the beginning of the sentence cannot attend to the verb or the later noun.
Future analysis should inspect voice and position separately and compare
interventions against unchanged and suitable control conditions.

This repository provides dataset preparation, role activation extraction,
training role means pooled across voices and separately for each voice, and
separate agent/patient interventions with counterfactual-answer logit evaluation.
The fixed agent question is evaluated on test and held-out-noun sentences;
unchanged baselines, zero updates, and final-block invariance provide controls.
