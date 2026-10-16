# Theory: agreement-stratified ranking certification (G2.1–G2.3)

Written 2026-09-19. Every proposition is derived here from its assumptions. The brute-force
checks in `tests/test_certification.py` confirm the implementation of each result. They are
not its proof.

## 0. Setting and notation

- Models $m\in[M]$ and items $i\in[N]$, where $N=1{,}324$ is MATH-Hard. A row is $(m,i)$.
- $K$ verifiers give verdict vector $v_{mi}\in\{0,1\}^K$ and **cell** $c(m,i)$.
  Concordant cells: $\mathcal C=\{0^K,1^K\}$. Discordant cells $\mathcal D$ are the rest.
- Human reference $Y^*_{mi}\in\{0,1\}$ (harness routine on the transcription; indeterminate
  counts as 0). It is observed only on audited rows.
- Census shares $s_{mc}=|\{i:c(m,i)=c\}|/N$ are known exactly.
- $p_{mc}$ is the mean of $Y^*$ over model $m$'s rows in cell $c$ (a finite-population
  quantity).
- **Estimand F (finite benchmark).** $A_m=\frac1N\sum_i Y^*_{mi}=\sum_c s_{mc}p_{mc}$ and
  $\Delta_{jk}=A_j-A_k=\sum_c (s_{jc}p_{jc}-s_{kc}p_{kc})$.
- **Estimand S (superpopulation).** The items are an i.i.d. draw from an item population
  $\mathcal P$, and $\delta_{jk}=\mathbb E_{\mathcal P}[Y^*_j(I)-Y^*_k(I)]$.
- **Audit design.** Stratified simple random sampling without replacement (SRSWOR).
  An *audit sub-population* $g$ is (design stratum × model). For the existing 400-item audit,
  the strata are (boxed stratum, subject, length quartile), sampled across models jointly. For
  the new audit they are (model, cell).
- We condition on the realized sample counts $n_{gc}$ in every (sub-population, cell). Given
  those counts, the labeled rows in each $(g,c)$ are an SRSWOR from that sub-population. This
  holds because conditioning an SRS on the counts falling in subsets leaves an SRS within
  each subset. The samples are also independent across $(g,c)$.
  $p_{gc}$ denotes the mean of $Y^*$ over $(g,c)$.

## 1. The heterogeneity parameter (G2.1)

**H(ε).** For each concordant cell $c\in\mathcal C$: $\max_g p_{gc}-\min_g p_{gc}\le\varepsilon$,
over all audit sub-populations $g$ with rows in $c$.

H(ε) restricts **cross-model (and cross-stratum) heterogeneity**, meaning the spread of
concordant correctness. It says nothing about the **absolute level**: $p_{1^K}$ may be 0.99
or 0.9. The level is estimated from the audit, and H(ε) bounds only how far any one model can
depart from the others. H(0) is the homogeneity assumption behind the earlier 84.

With a (model, cell)-stratified design, $g$ is the model and H(ε) is a statement about
models. With the old design, H(ε) also covers the subject and length strata within a model.
That is stronger, and it is the version the old audit actually needs.

**H(ε) is about realized fractions, so ε=0 rarely holds exactly.**
$p_{gc}$ is a fraction of finitely many fixed responses. Even if every model had the same
*propensity* $\pi$ of human-judged correctness in a concordant cell, the realized fractions
would spread by about $\sqrt{\pi(1-\pi)/N_{gc}}$ across sub-populations. Equal propensities do
not imply equal realized fractions, so H(0) can fail even in this case, the one most favourable
to pooling.

The coverage simulation (§8) shows this directly. With homogeneous propensities, analysis
at ε=0 can exceed its nominal error, and analysis at a small ε>0 restores it. Consequences:
- ε=0 counts (the earlier 84, and the ε=0 row of `certify_results.json`) are an idealization.
  They are reported as the upper end of the ε curve, never as the headline.
- The design matters. With (stratum × model) sub-populations (old audit), $N_{gc}$ is small
  and the natural spread is large. A (model × cell)-stratified audit (G3) makes $g$ the
  model, so $N_{gc}$ is large and the spread small.

**Pooled parameter.** $\tilde p_c=\sum_g w_{gc}\,p_{gc}$ with sampling weights
$w_{gc}=n_{gc}/n_c$. It is the expectation of the pooled audit proportion $x_c/n_c$.

### 1.1 Derived reference ε_ref(a) (replaces v1's ε_ref)

For $c=0^K$ let the error be $e_g=p_{gc}$. For $c=1^K$ let it be $e_g=1-p_{gc}$. Either way
$e_g\ge0$. On the event $E_c=\{\tilde e_c\le u_c\}$, where $u_c$ is the upper confidence
bound of the pooled error:

$$w_{gc}\,e_g\le\sum_{g'}w_{g'c}e_{g'}=\tilde e_c\le u_c\;\Rightarrow\;e_g\le u_c/w_{gc}.$$

Since every $e\ge0$, the range is at most $\max_g e_g$. So

$$\varepsilon\;\le\;\varepsilon_{\text{ref}}^{(a)}=\max_{c\in\mathcal C}\;u_c\big/\min_{g:\,n_{gc}>0}w_{gc}.$$

The bound holds only over **sampled** sub-populations. A sub-population with no labels in $c$
is unconstrained. The bound needs no homogeneity assumption; it is weight-dependent and
usually weak, and that weakness is itself informative. It is reported as a number.

**(b)** After the new audit: $\max_m$ of the per-model exact upper bounds on $e_{mc}$.
**(c)** Preregistered scientific tolerance: $\varepsilon_0=0.01$. This is declared as a
judgement, before results, and is not a measurement.

The primary presentation is the full curve of certified orderings against ε, with (a)–(c)
marked on it. No ε is chosen for its count.

## 2. Lemma 1 — design-valid bound for a heterogeneous pooled cell (G2.3)

**Added 2026-09-25 — random label counts.** The claim below takes each $n_{gc}$ as fixed.
The existing 400-item audit was stratified on (boxed-verifier stratum, subject), not on the
K=3 cells, so its $n_{gc}$ are random. The claim holds conditionally on the realised counts:
within a stratum the draw is a uniform random subset of fixed size, and conditional on the
domain counts the per-domain samples are independent uniform subsets of their domains (the
uniform law over subsets with fixed domain counts factorises). With $g$ = model × design
stratum, the $x_{gc}$ are then independent hypergeometric given $\{n_{gc}\}$, and $\tilde p_c$
uses the realised weights. Coverage holds for every realisation, hence unconditionally; P1
needs $\tilde p_c$ only to be a convex combination of the $p_{gc}$, which it always is.
Checked by `tests/test_certification.py::test_kl_interval_covers_with_random_domain_counts`.

**Claim.** Let $x_c=\sum_g x_{gc}$, with independent $x_{gc}\sim$
Hypergeometric$(N_{gc},N_{gc}p_{gc},n_{gc})$. Then for $t>0$:

$$\Pr(x_c/n_c\ge\tilde p_c+t)\le e^{-n_c\,\mathrm{KL}(\tilde p_c+t\,\|\,\tilde p_c)},\qquad
\Pr(x_c/n_c\le\tilde p_c-t)\le e^{-n_c\,\mathrm{KL}(\tilde p_c-t\,\|\,\tilde p_c)}.$$

**Proof.**
1. Hoeffding (1963, Thm 4): for continuous convex $f$, the sum of an SRSWOR is dominated in
   $\mathbb E f$ by the sum of a with-replacement sample. Hence
   $\mathbb E e^{\lambda x_{gc}}\le(1-p_{gc}+p_{gc}e^\lambda)^{n_{gc}}$.
2. Independence across $g$ lets the moment generating functions multiply.
3. $p\mapsto\log(1-p+pe^\lambda)$ is concave, so by Jensen with weights $w_{gc}$:
   $\prod_g(1-p_{gc}+p_{gc}e^\lambda)^{n_{gc}}\le(1-\tilde p_c+\tilde p_c e^\lambda)^{n_c}$.
4. The Chernoff bound optimized over $\lambda$ gives the KL exponent. ∎

**Interval.** $\mathrm{KL}_\alpha(x,n)=\{q:\,n\,\mathrm{KL}(\hat p\|q)\le\log(2/\alpha)\}$ has
coverage $\ge1-\alpha$ for $\tilde p_c$. It needs no identical rates, no replacement, and no
asymptotics. At $x=0$ its upper end is $1-(\alpha/2)^{1/n}$, the same as Clopper–Pearson.
**Headline numbers use KL.** Clopper–Pearson is reported alongside as the tighter,
assumption-dependent alternative: it is exact only for identical rates with replacement.

## 3. P1 — identified range of Δ (sharp at ε=0, conservative for ε>0)

**Parameterization.** $\Delta_{jk}=\sum_c(s_{jc}p_{jc}-s_{kc}p_{kc})$ is linear in the
$p$'s and separable across cells.

**Feasible region.** On the event $\tilde p_c\in[L_c,U_c]$ for each $c\in\mathcal C$, and
under H(ε), the pair $(p_{jc},p_{kc})$ lies in

$$R_c(\varepsilon)=\{(p_j,p_k)\in[\ell,h]^2:\ |p_j-p_k|\le\varepsilon\},\quad
\ell=\max(0,L_c-\varepsilon),\ h=\min(1,U_c+\varepsilon).$$

*Necessity.* $p_{jc}$ and $p_{kc}$ are convex combinations of their own sub-population rates
(weights $N_{gc}/N_{mc}$), and $\tilde p_c$ is a convex combination of all of them. Under
H(ε) all sub-population rates lie in some interval $I$ of width ε, so
$p_{jc},p_{kc},\tilde p_c\in I$. That gives $|p_{jc}-p_{kc}|\le\varepsilon$ and
$|p_{mc}-\tilde p_c|\le\varepsilon$, and with $\tilde p_c\in[L_c,U_c]$ the box follows.
Clipping to [0,1] is the logical range.

For a discordant cell, $p_{mc}\in[l_{mc},u_{mc}]$ independently per model: [0,1] if the
cell is unaudited, else a per-model design-valid KL interval.

**Closed form.** Let $a=s_{jc}\ge0$ and $b=s_{kc}\ge0$. Then

$$\min_{R_c}(a p_j-b p_k)=\begin{cases}a\ell-b\min(h,\ell+\varepsilon)&a\ge b\\ a\max(\ell,h-\varepsilon)-b h&a<b\end{cases}$$

*Proof.* For fixed $p_j$, the optimum is $p_k=\min(h,p_j+\varepsilon)$. The resulting
$g(p_j)=ap_j-b\min(h,p_j+\varepsilon)$ is convex and piecewise linear, with slope $a-b$ below
$h-\varepsilon$ and slope $a$ above it. If $a\ge b$ it is nondecreasing, so the minimum is at
$\ell$. Otherwise the minimum is at the kink $h-\varepsilon$, clipped to $\ell$. When the clip
binds, $\ell+\varepsilon>h$ gives $\min(h,\ell+\varepsilon)=h$. ∎
The maximum follows by symmetry: $\max(ap_j-bp_k)=-\min(bp_k-ap_j)$ with the roles swapped.

**Range.** $[\mathrm{LB},\mathrm{UB}]_{jk}(\varepsilon)=\sum_{c\in\mathcal C}[\min,\max]_{R_c}
+\sum_{c\in\mathcal D}[s_{jc}l_{jc}-s_{kc}u_{kc},\ s_{jc}u_{jc}-s_{kc}l_{kc}]$.
The objective is linear over a convex polygon, so the extremes are at vertices.

*Sharp or conservative.*
- **Sharp at ε=0.** $R_c(0)=\{p_j=p_k=q\in[L_c,U_c]\}$, and every such $q$ is attained by
  setting all sub-population rates to $q$. This is the earlier "shared" analysis.
- **Conservative (an outer region) for ε>0.** The weights $w_{jc},w_{kc}$ can stop $\tilde p$
  from reaching $[L_c,U_c]$ for some points of $R_c$. We do not claim sharpness there.

## 4. P2 — breakdown value

**Certification.** Ordering $j\succ k$ is certified at ε if
$\mathrm{LB}_{jk}(\varepsilon)-h_{jk}(\varepsilon)>0$, where $h_{jk}$ is the sampling
half-width (§6); $h\equiv0$ for estimand F.

**Monotonicity.** $R_c(\varepsilon)$ grows with ε. So $\mathrm{LB}$ is continuous,
piecewise linear and nonincreasing, and $h_{jk}$ is nondecreasing (§6).

**Breakdown value.** $\varepsilon^*_{jk}=\sup\{\varepsilon:\mathrm{LB}_{jk}(\varepsilon)>h_{jk}(\varepsilon)\}$
is well defined and computed by bisection to $10^{-9}$.

**Slope, derived.** Within the interior (no clipping):
- if $a\ge b$, $\frac{d}{d\varepsilon}\big[a(L-\varepsilon)-bL\big]=-a$;
- if $a<b$, $\frac{d}{d\varepsilon}\big[a U-b(U+\varepsilon)\big]=-b$.

So each concordant cell contributes slope $-\max(s_{jc},s_{kc})$, **not**
$-(s_{jc}+s_{kc})$.

**Clipping binds immediately in practice.** For $c=0^K$, $L_c=0$, so $\ell=0$ for every
$\varepsilon>0$. With $a\ge b$ the cell's term becomes $-b\min(h,\varepsilon)$, slope $-b$,
until $\varepsilon=h$. For $c=1^K$, $U_c\approx1$ clips $h$ at 1 once
$\varepsilon\ge1-U_c$. The implementation evaluates the exact clipped closed form, never the
interior slope.

**v1 formula, checked.** $\mathrm{LB}(0)-\varepsilon\sum_c(s_{jc}+s_{kc})$ is the relaxation
"each $p_{mc}$ within ε of a common $q$". It drops $|p_j-p_k|\le\varepsilon$ and ignores
clipping. It is valid but looser, so it is not used.

**Exact or lower bound.** $\varepsilon^*$ is exact for the outer region $R_c$. It is therefore
a **lower bound** on the breakdown value of the sharp identified set: conservative, and it
never overstates robustness.

**Double counting.** The audit term bounds $\Delta$ (F), conditional on the realized
benchmark. The sampling term bounds $\delta-\Delta$, which comes from the item draw. These are
distinct error sources, and the union bound adds their levels:
$\Pr(\delta\notin[\mathrm{LB}-h,\mathrm{UB}+h])\le\Pr(\Delta\notin[\mathrm{LB},\mathrm{UB}])+\Pr(|\delta-\Delta|>h)$.
Nothing is double-counted. The combination is conservative because it adds levels instead of
convolving the two errors.

## 5. P3 — one verifier under a common mechanism

**PV parameterization** (what an audit stratified on one verifier identifies,
CORE_ESTIMAND_AUDIT.md): common predictive values $\pi_1=\Pr(Y^*=1\mid v=1)$ and
$\pi_0=\Pr(Y^*=1\mid v=0)$ for all models. Then $A_m=\pi_0+a_m(\pi_1-\pi_0)$ with
$a_m=s_{m,1}$, so

$$\Delta_{jk}=(\pi_1-\pi_0)(a_j-a_k).$$

**Error-rate parameterization**: common $q=\Pr(v=1\mid Y^*=0)$ and
$r=\Pr(v=0\mid Y^*=1)$. Then $a_m=q+(1-q-r)A_m$, so

$$a_j-a_k=(1-q-r)\,\Delta_{jk}.$$

(Plan v1 wrote $\Delta^*=(a_i-a_j)(1-q-r)$. The factor is on the other side. The conclusion
does not change.)

**Consequence.** All pairwise differences are scaled by one common factor. A common error
mechanism **cannot produce pair-specific reorderings**. Its only global effect is a sign flip
of every ordering, when $\pi_1<\pi_0$ or $q+r>1$; at $\pi_1=\pi_0$ or $q+r=1$ nothing is
identified.

The observed pair-specific reversals between verifiers therefore imply model-specific
mechanisms. The one-verifier pooled analysis is H(0) with $K=1$: pooling covers the whole
population. The $K$-verifier analysis restricts pooling to the concordant cells, and the
discordant cells stay model-specific.

## 6. Sampling half-width (estimand S)

Let $D_i=Y^*_j(i)-Y^*_k(i)$. Then $\mathrm{Var}_{\mathcal P}(D)\le\Pr(D\ne0)$.

The benchmark analogue $f_{jk}=\frac1N\sum_i\mathbf 1[Y^*_{ji}\ne Y^*_{ki}]$ is bounded
without observing $Y^*$. For any verifier $v$:

$$\mathbf 1[Y^*_j\ne Y^*_k]\le\mathbf 1[v_j\ne v_k]+\mathbf 1[Y^*_j\ne v_j]+\mathbf 1[Y^*_k\ne v_k],$$

so $f_{jk}\le d^v_{jk}+e^v_j+e^v_k$. Here $d^v_{jk}$ is observed, and
$e^v_m=\sum_c s_{mc}\,[v_c{=}1](1-p_{mc})+[v_c{=}0]\,p_{mc}$ is bounded above using the
identified bounds on $p_{mc}$ from the same event, at the same ε.

With $\bar f_{jk}(\varepsilon)=\min\!\big(1,\min_v(d^v_{jk}+\bar e^v_j+\bar e^v_k)\big)$, the
primary half-width is $h_{jk}=z_{1-\alpha_s/2}\sqrt{\bar f_{jk}/N}$: paired, conservative, and
**asymptotic** (CLT). This replaces the earlier unpaired plug-in from one verifier's accuracy,
which is kept only as `legacy` for the regression anchor.

Finite-sample sensitivity is Hoeffding over $D\in[-1,1]$:
$h=\sqrt{2\log(2/\alpha_s)/N}$, which is distribution-free and very conservative.

## 7. P4 — falsification power

A (model, concordant cell) sub-population with error rate $\rho$, audited with $n$ SRSWOR
labels, shows at least one error with probability $\ge1-(1-\rho)^n$. The reason:
$\binom{N-R}{n}/\binom Nn\le(1-R/N)^n$, so sampling without replacement only helps.

Hence $n\ge\log(1-\pi)/\log(1-\rho)$ gives power $\ge\pi$. For ρ=0.10 and π=0.8 this is
n=16; for ρ=0.20 it is n=8.

**Caution for G3.5.** "At least one error" is necessary for rejection, not sufficient. The
rejection rule must be a valid test of H(ε₀): reject if
$\max_m \mathrm{LB}(e_{mc})-\min_m \mathrm{UB}(e_{mc})>\varepsilon_0$ with simultaneous
intervals. Its power is lower than the P4 figure and is computed exactly in G3.1.

## 8. Coverage (G2.3)

Design elements: stratified SRSWOR, a finite population, pooled concordant cells,
per-model discordant cells, 351 pairs, and benchmark sampling.

- **Per ordering.** The audit part uses level 0.975, Šidák across the independently
  estimated parameters the ordering uses: two pooled concordant cells plus the audited
  discordant cells of $j$ and $k$. Each parameter gets a KL interval (Lemma 1). The sampling
  part uses 0.975 (§6). The union bound gives ≥ 0.95.
- **Simultaneous (family of 351).** The pooled concordant event is shared by all pairs.
  Discordant parameters take Šidák over all audited per-model parameters, and sampling takes
  Bonferroni $\alpha_s/351$.
- **Rank intervals.** If every certified claim is true, model $m$'s true rank lies in
  $[1+\#\{k\succ m\},\,M-\#\{m\succ k\}]$. So the simultaneous rank intervals inherit the
  family-wise level.

**Proposition C.**
- *Under H(ε), estimand F:* all audit-only claims are simultaneously correct with probability
  $\ge1-\alpha_a$, **exactly in finite samples** (Lemma 1, independence, Šidák).
- *Estimand S:* $\ge1-\alpha_a-\alpha_s$ **asymptotically**, the sampling part by the CLT with
  a conservative variance.

Simulation (`scripts/fb_certify_coverage_sim.py`, ≥2,000 replicates, 27 models, 351 pairs)
checks the tightness of these statements. It is not their proof.

**Simulation outcome** (`results/flipbudget/certify_coverage_sim.json`):
- **Validity.** In every scenario where finite H(ε) holds (K ∈ {2,3}, ε ∈ {0, 0.01}), no
  certified claim was wrong: audit-only, combined per-ordering, and combined simultaneous.
  The procedure is valid there, and **conservative, not tight**. Its realized error sits far
  below nominal, mainly because unaudited discordant cells enter at worst case [0,1] and the
  paired variance bound adds the error terms.
- **Misspecification.** When the truth has ε=0.02 and the analysis uses ε=0, the audit-only
  family-wise error exceeds nominal. This is the failure mode the ε curve and breakdown values
  exist to expose.
- **Natural noise.** Homogeneous propensities push K=2 audit-only claims past nominal at
  ε=0, and ε=0.01 restores validity. Combined claims stayed at zero error in every scenario,
  because the sampling half-width is wide enough to absorb the realized-fraction noise.
