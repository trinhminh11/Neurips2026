# Trajectory-Centric Skill Discovery Beyond DIAYN

## 1. Motivation

DIAYN learns skills by making the skill variable identifiable from states, together with a maximum-entropy policy. This is a strong baseline, but its implicit notion of a skill is still largely state-centric: a skill is useful if visited states reveal which latent `z` generated them. In practice, this can miss an important distinction:

- two skills may reach similar states but follow very different trajectories;
- for downstream control, the semantic meaning of a skill is often **how the agent moves**, not only **where it ends up**.


> DIAYN can be trajectory-blind: distinct behaviors may be weakly separated if they pass through overlapping state regions or end in similar states.

This motivates a trajectory-centric objective.

## 2. Main Proposal

Let

- $Z$ be the skill latent,
- $S_t$ be the current state,
- $H_t^k := (S_{t}, S_{t+1}, ..., S_{t+k-1})$ be a local future trajectory segment.

Instead of maximizing skill-state mutual information, I propose maximizing

$I(Z; H_{t+1}^{k-1} | S_t)$.

Interpretation:

- after fixing the current state $S_t$,
- the next local trajectory segment $H_{t+1}^{k-1}$
- should be predictive of which skill $Z$ is active.

This matches the intended skill semantics better: a skill is defined by its local behavioral pattern from the same starting context, not just by a final state label.

## 3. Why This Is Better Aligned With Skills

This objective captures the idea that:

- different skills may still reach the same region of the state space;
- what should distinguish them is the local trajectory they generate;
- therefore, skill meaning should be attached to trajectory style or control mode rather than isolated states.

Examples:

- one skill reaches a point by moving left first, then forward;
- another reaches a similar point by moving right first, then forward.

A state-only objective may not separate them well. A trajectory-conditioned objective can.

## 4. Idea 1: Trajectory MI With a Surrogate Reward

This is the first idea, and it should be treated as a separate paper direction from Idea 2 below.

For clarity, in the rest of this note I use

$$
H_t^k := (S_{t}, S_{t+1}, \dots, S_{t+k-1})
$$

for the future-only trajectory segment. Then the objective of Idea 1 is

$$
I(Z; H_{t+1}^{k-1} \mid S_t).
$$

### 4.1. Exact Identity

The clean starting point is

$$
\begin{aligned}
I(Z; H_{t+1}^{k-1} \mid S_t) &= I(H_{t+1}^{k-1}; Z \mid S_t)\\
&= \mathcal{H}[Z \mid S_t] - \mathcal{H}[Z \mid S_t, H_{t+1}^{k-1}] \\
&= \mathcal{H}[Z \mid S_t] - \mathcal{H}[Z \mid H_{t}^{k}] \\
&= \mathbb{E}\left[\log p(z \mid H_{t}^{k}) - \log p(z \mid S_t)\right].
\end{aligned}
$$

This identity is exact.

It is important **not** to replace $p(z \mid S_t)$ with the prior $p(z)$ unless additional assumptions justify it. After the agent has already executed part of the skill, the current state generally contains information about $z$, so in general

$$
p(z \mid S_t) \neq p(z).
$$

### 4.2. A Useful Chain-Rule View

The same quantity admits the conditional chain-rule decomposition

$$
I(Z; H_{t+1}^{k-1} \mid S_t) = I(H_{t+1}^{k-1}; Z \mid S_t) \\
= \sum_{i=1}^{k} I\!\left(S_{t+i}; Z \mid H_{t}^{t+i}\right).
$$

This supports the trajectory interpretation nicely:

- each newly observed state can contribute additional evidence about the active skill;
- the skill is defined by how information accumulates along the rollout;
- a long-horizon objective can be approximated through smaller local terms.

### 4.3. Surrogate Reward

Idea 1 then uses a simple learned surrogate:

- a trajectory discriminator $q_{\phi}(z \mid H_{t}^{k})$,
- a state-only baseline $q_{\psi}(z \mid S_t)$.

The surrogate score is

$$
J_{\mathrm{sur}}
= \mathbb{E}\left[\log q_{\phi}(z \mid H_{t}^{k}) - \log q_{\psi}(z \mid S_t)\right].
$$

The corresponding intrinsic reward is the same sample-wise quantity

$$
r_t^{\mathrm{sur}}
= \log q_{\phi}(z \mid H_{t}^{k}) - \log q_{\psi}(z \mid S_t).
$$

Interpretation:

- reward a trajectory segment if it makes the skill easier to infer than the current state alone;
- penalize solutions where the future trajectory adds little information beyond $S_t$.

This is the more novel direction, but it should still be described honestly:

> Idea 1 is a trajectory-MI-inspired surrogate objective, not by itself a proven variational lower bound.

### 4.4. Incremental Information-Gain Variant

If you want a denser reward within Idea 1, one natural heuristic is

$$
r_{t}^{\mathrm{IG}}
= \log q_{\phi}(z \mid H_{t}^{k}) - \log q_{\phi}(z \mid H_{t}^{k-1}),
$$

where $H_{t}^{k} := (S_{t}, \dots, S_{t+k-1})$.

This says:

- if the newest state makes the skill much easier to identify, reward it more;
- if it adds little information, reward it less.

This is a useful heuristic for Idea 1, but I would still present it as reward shaping rather than as a theorem.

## 5. Idea 2: Variational Conditional InfoNCE

This is a second, separate paper direction. Unlike Idea 1, the central objective here is a **proved variational lower bound** on conditional mutual information, following the variational conditional InfoNCE result of Sordoni et al.

Again let the context be $S_t$, the future segment be $H_{t+1}^{k-1}$, and the skill be $Z$. Then the target quantity is

$$
I(Z; H_{t+1}^{k-1} \mid S_t) = I(H_{t+1}^{k-1}; Z \mid S_t)
$$

### 5.1. Proven Objective

One variational conditional InfoNCE objective is

$$
I(H_{t+1}^{k-1}; Z \mid S_t) \ge I_{\mathrm{VAR}}(H_{t+1}^{k-1}, Z|S_t, \theta,\phi, K),
$$

with

$$
\begin{aligned}
I_{\mathrm{VAR}}(H_{t+1}^{k-1}, Z|S_t, \theta,\phi, K)
=\;&
\mathbb{E}\left[
\log
\frac{
\exp\!\left(f_{\theta,\phi}(S_t,H_{t+1}^{k-1}, Z^{+})\right)
}{
\frac{1}{K}\sum_{j=1}^{K}
\exp\!\left(f_{\theta,\phi}(S_t,H_{t+1}^{k-1}, Z_j)\right)
}
\right] \\
&\quad - \mathbb{E}_{S_t}\!\left[
D_{\mathrm{KL}}\!\left(p(Z \mid S_t)\,\|\,q_{\phi}(Z \mid S_t)\right)
\right].
\end{aligned}
$$

Here:

- $(S_t, H_{t+1}^{k-1}, Z^{+})$ is a positive tuple sampled from the true joint distribution;
- $Z_j \sim q_{\phi}(Z \mid S_t)$ are the negative skill samples conditioned on the same context;
- $q_{\phi}(Z \mid S_t)$ is the variational approximation to the in-context skill distribution.

A standard compatible parameterization is

$$
f_{\theta,\phi}(S_t, H_{t+1}^{k-1}, z)
= \log \frac{g_{\theta}(z \mid S_t, H_{t+1}^{k-1})}{q_{\phi}(z \mid S_t)} + c(S_t, H_{t+1}^{k-1}),
$$

where $c(S_t, H_{t+1}^{k-1})$ does not depend on $z$.

This is exactly the right place to be strong:

- this is a **proved lower bound** on the conditional mutual information;
- if $q_{\phi}(Z \mid S_t) = p(Z \mid S_t)$, it reduces to conditional InfoNCE;
- as the approximation improves, the gap introduced by the KL term shrinks.

### 5.2. Reward for Idea 2

If Idea 2 is treated as its own method, then the intrinsic reward should be the sample contribution of the same variational objective itself.

Write the reward as

$$
r_t^{\mathrm{VAR}}
=\log\frac{\exp\!\left(f_{\theta,\phi}(S_t, H_{t+1}^{k-1}, z_t)\right)}{\frac{1}{K}\sum_{j=1}^{K}\exp\!\left(f_{\theta,\phi}(S_t, H_{t+1}^{k-1}, z_j)\right)}- D_{\mathrm{KL}}\!\left(p(Z \mid S_t)\,\|\,q_{\phi}(Z \mid S_t)\right),
$$

where $z_t$ is the skill that generated the sampled trajectory segment and $z_j \sim q_{\phi}(Z \mid S_t)$ are the negatives.

So for Idea 2, the message is simple:

- the reward is not a heuristic posterior-difference term;
- the reward is the minibatch/sample form of the **proved variational conditional InfoNCE objective**.

### 5.3. Interpretation

Idea 2 says that a good skill should:

- score highly against negative skills drawn from the same state-conditioned skill distribution;
- remain close to the true conditional prior through the KL correction;
- maximize a formal lower bound on $I(Z; H_{t+1}^{k-1} \mid S_t)$.

Compared with Idea 1, this version is mathematically cleaner, but it also overlaps more with prior contrastive skill-discovery work.

## 6. Novelty Assessment

### Compared with DIAYN

The difference is real and meaningful:

- DIAYN emphasizes skill identifiability from states;
- this proposal emphasizes skill identifiability from **local future trajectories conditioned on the current state**.

That is a defensible conceptual shift.

### Compared with CIC

This is the main novelty risk.

CIC already maximizes mutual information between skills and state-transitions using contrastive learning. In its basic form, the contrastive view is much closer to a one-step signal such as $(S_t, S_{t+1})$. Your proposal is stronger when it is written around a **multi-step conditional future segment** $(S_{t+1}, \dots, S_{t+k})$ with $k > 1$.

So if the proposal is written as

> "Use InfoNCE to maximize MI between skill and transition,"

then it will likely look too close to prior work.

To stay novel, the key message should be:

- not generic one-step transition-skill contrast,
- but **conditional local trajectory identifiability**,
- with `S_t` as context,
- with a future window `(S_{t+1}, ..., S_{t+k})` rather than only `(S_t, S_{t+1})`,
- and with an information-gain / decomposition view over trajectory prefixes.

That angle is more distinct. Put differently:

- if `k = 1`, the comparison to CIC becomes much tighter;
- if `k > 1` and the method really exploits multi-step prediction or prefix-wise information gain, the proposal is easier to defend as a different idea.

### Compared with DUSDi

DUSDi focuses on disentangled skill components and downstream compositionality. Your idea is largely orthogonal:

- DUSDi asks: can different latent factors control different parts of the environment?
- your idea asks: can a skill be defined by a trajectory pattern rather than state occupancy?

So the overlap is limited.

## 7. My Recommendation

I would now present these as **two separate ideas**, not one combined paper.

For a NeurIPS-style novelty pitch, I prefer **Idea 1**:

> Learn skills by maximizing conditional mutual information between a skill latent and a local future trajectory segment given the current state, so that skills represent distinct behavioral patterns rather than only distinct visited states.

Why I prefer Idea 1:

- the trajectory-centric semantics are more distinctive relative to DIAYN;
- the information-gain view over prefixes is intuitive and easy to explain;
- it is the stronger novelty claim.

For a more theory-clean paper, **Idea 2** is stronger:

- the main objective is a proved variational lower bound;
- the reward is directly tied to that exact objective;
- but the contrastive framing is closer to CIC and related work.

So my recommendation is:

- if you want the more novel story, pursue Idea 1;
- if you want the more formal and technically grounded story, pursue Idea 2;
- do not present them as one paper unless you later discover a genuinely unified theorem and algorithmic benefit.

## 8. Strengths

- Clear intuition: skills should encode behavior, not only endpoints.
- Easy to explain to colleagues and reviewers.
- Compatible with hierarchical RL, because a high-level controller can pick among distinct trajectory modes.
- Naturally supports denser intrinsic rewards through local information gain.

## 9. Honest Limitations

- The conditional baseline `p(z | S_t)` is the hard part; it cannot simply be ignored.
- If `k` is too small, the objective may collapse to a short-horizon transition discriminator.
- If `k` is too large, optimization becomes harder and data efficiency may suffer.
- The novelty claim is strongest when the paper emphasizes conditional trajectory semantics and decomposition, not just contrastive estimation.

## 10. Suggested Pitches

### Idea 1 Pitch

We propose a trajectory-centric alternative to DIAYN in which a skill is defined not only by the states it visits but by the local trajectory pattern it induces from the same starting state. Formally, we maximize conditional mutual information $I(Z; H_{t+1}^{k-1} \mid S_t)$ between the skill latent $Z$ and a future trajectory segment $H_{t+1}^{k-1}$, conditioned on the current state $S_t$. This objective encourages skills to encode distinct behavioral modes rather than merely distinct state occupancies. In practice, we optimize a surrogate reward based on the posterior improvement from $S_t$ to $(S_t, H_{t+1}^{k-1})$, optionally with incremental information-gain rewards over trajectory prefixes.

### Idea 2 Pitch

We propose a conditional contrastive skill-discovery objective based on the proved variational conditional InfoNCE lower bound. With $S_t$ as context, $H_{t+1}^{k-1}$ as a future trajectory segment, and $Z$ as the skill latent, we maximize a formal lower bound on $I(Z; H_{t+1}^{k-1} \mid S_t)$ using state-conditioned negatives and a variational approximation $q_{\phi}(Z \mid S_t)$. The intrinsic reward is the sample contribution of this exact variational objective, yielding a mathematically clean training signal for learning skills that are identifiable from local trajectories.

## References

- DIAYN: [https://arxiv.org/abs/1802.06070](https://arxiv.org/abs/1802.06070)
- CIC: [https://arxiv.org/abs/2202.00161](https://arxiv.org/abs/2202.00161)
- DUSDi: [https://arxiv.org/abs/2410.11251](https://arxiv.org/abs/2410.11251)
- Decomposed / variational conditional MI estimation context: [https://proceedings.mlr.press/v139/sordoni21a.html](https://proceedings.mlr.press/v139/sordoni21a.html)
