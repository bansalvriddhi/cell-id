# What the linear pilot can identify

This is a derivation of standard linear-Gaussian design identities used in the pilot, not a claim of a new theorem.

For a context latent vector z of dimension r, the response to probe p is y_p = mu_p + A_p z + epsilon_p, with prior z ~ N(0,I_r) and independent observation/model-discrepancy noise epsilon_p ~ N(0,tau^2 I_G). A_p is a G-by-r matrix learned from other contexts. Given a probe set S, define

    Lambda_S = I_r + (1/tau^2) sum_{p in S} A_p^T A_p
    Sigma_S = Lambda_S^{-1}
    m_S = Sigma_S (1/tau^2) sum_{p in S} A_p^T (y_p - mu_p).

Then E[y_q | S] = mu_q + A_q m_S for an unqueried perturbation q.

## Why probe count alone is insufficient

If a vector v lies in the null space of every selected A_p, then Lambda_S v = v: those experiments supply no information about that latent direction. In a noiseless model the stacked selected matrices identify z exactly only if their column rank is r. A single transcriptomic experiment returns many gene measurements, so there is no general lower bound requiring r distinct drugs; one A_p can already have rank r. Conversely, many redundant drugs can leave directions unobserved. In noisy data, singular values and model mismatch matter, not rank alone.

## D-optimal information gain

Under the stated Gaussian assumptions, latent mutual information is

    I(z; y_S) = 0.5 log det(Lambda_S).

The incremental score for p is half of log det(Lambda_S + A_p^T A_p/tau^2) minus log det(Lambda_S). Dropping the constant half leaves the same selected probe. This is D-optimal design under a fixed learned prior; calling it a newly invented joint optimizer would be incorrect. Greedy selection constructs a sequence and is not an exact global maximization over every possible size-K set.

## Whole-panel predictive-variance reduction

Let T be a fixed set of target perturbation identities and Q_T = sum_{q in T} A_q^T A_q. The reducible latent contribution to total squared prediction uncertainty is

    R_T(S) = tr(Q_T Sigma_S).

Adding p reduces this by tr(Q_T (Sigma_S - Sigma_{S union {p}})). The implemented V-optimal selector maximizes this score. It targets expected squared prediction error under the assumed model. It is neither the same as D-optimal latent entropy nor necessarily the entropy of all predicted gene outputs. Noise terms that remain constant across probe choices do not affect its argmax.

## Outcome independence and model mismatch

Sigma_S depends on selected identities, the training-derived A matrices and tau, but not on measured y values. Thus these policies do not change future selections after seeing surprising outcomes. Their predictions do change through m_S. A nonlinear or outcome-adaptive PRISM selector is a future extension.

All information and variance claims above are conditional on a correctly specified basis and noise model. A smaller model posterior covariance does not establish that real biological prediction error decreases, or that confidence intervals are calibrated. The real held-out-context experiment is the test of that mismatch. No dynamical trajectories, control laws or clinical effects follow from this static model.
