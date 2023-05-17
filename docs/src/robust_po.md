# Robust Portfolio Optimization
**Acknowledgements**: Robust Formulations based on and inspired by Professor [Davi M. Valladão](http://www.ind.puc-rio.br/en/equipe/davi-michel-valladao/)'s lectures on "Capital Market".

## Motivation
Portfolio Optimization (PO) formulations were developed to adapt to a variety of settings for decisions under uncertainty. These formulations depend on the available information of the uncertain data and the risk aversion of the decision-maker.

The available information for many market data (e.g. stock prices, future prices, etc) have non-stationary profiles that are hard to make statistical inference on. Thus, it can be hard to find good probability distributions needed for stochastic optimization. 

One alternative approach is Robust Optimization.

## Background
Robust Optimization (RO) problems belong to the class of optimization under uncertainty problems where some problem data is uncertain (either because the decision is taken before the realization of the random event or because its observation is not available). Usual application cases for RO are when there isn’t sufficient information to derive probability distributions, but this isn’t strictly necessary. RO focuses on guaranteeing solution feasibility for any possible value of the uncertain data inside a defined Uncertainty Set. In the case where the uncertainty impacts the objective function, it guarantees optimality for the works case scenario considered in the Uncertainty Set.

Many uncertainty sets have been proposed to accommodate different levels of conservatism and data structures ([1]-[7]). A comparison of uncertainty sets to usual risk measures used in finance are included in [4] and [11].

A collection of recent contributions to robust portfolio strategies are outlined in [7 - 10]. Data-driven approaches to robust PO has also gained interest in recent years and can be found in [12] - for a portfolio of stocks - and [13] - for a portfolio of future contracts. The results of these studies indicate promising alternatives for the integration of uncertain data in PO.

## Problem Definition
A simple version of the Mean-Variance Portfolio Allocation with robust uncertainty around the estimated mean returns (posed as a quadratic convex problem):

```math
\begin{aligned}
    \max_{w} \quad & R \\
    s.t. \quad & R = (\min r'w \; | \; r \in \Omega) \\
    & w ' \Sigma w \leq V_0 * W_0\\
    & w \in \mathcal{X} \\
\end{aligned}
```

Maximizes the worst case portfolio return (``R``) and limits the portfolio variance to a maximal risk parameter (``V_0``) normalized by current wealth (``W_0``).

Where ``\\mathcal{X}`` represents the additional constraints defined in the model by the user (e.g. a limit on maximum invested money).

A julia object representing this problem can be instanciated by the following command:

```julia
formulation = PortfolioFormulation(MAX_SENSE,
    ObjectiveTerm(ExpectedReturn(Ω)),
    RiskConstraint(Variance(d), LessThan(V_0 * W_0)),
)
```

where `d` is a `Sampleable` containing the estimated `\\Sigma` matrix.

### Bertsimas's Uncertainty Set
The uncertainty set proposed by Bertsimas in [6] is defined by the julia type ([`BudgetSet`](@ref)):

```math
\Omega = \left\{ \mu \; \middle| \begin{array}{ll}
s.t.  \quad \mu_i \leq \hat{r}_i + z_i \Delta_i \quad \forall i = 1:\mathcal{N} \\
\quad \quad \mu_i \geq \hat{r}_i - z_i \Delta_i  \quad \forall i = 1:\mathcal{N} \\
\quad \quad z_i \geq 0 \quad \forall i = 1:\mathcal{N} \\
\quad \quad z_i \leq 1 \quad \forall i = 1:\mathcal{N} \\
\quad \quad \sum_{i}^{\mathcal{N}} z_i \leq \Gamma \\
\end{array}
\right\} \\
```

where:
- ``\hat{r}``: Predicted mean of returns.
- ``\Delta``: Uncertainty around mean.
- ``\Gamma``: Budget (sometimes interpreted as number of assets in worst case).
- ``\Sigma``: Predicted covariance of returns.

The equivalent JuMP expression defining the worst case return (``R``) considering this uncertainty set can be constructed by the function `calculate_measure!(measure::ExpectedReturn{BudgetSet,WorstCase}, w)`. In this case, ``R`` in the described uncertainty set is defined by the following primal problem:  

```math
\begin{aligned}
\min_{\mu, z} \quad & \mu ' w \\
s.t. \quad & \mu_i \leq \hat{r}_i + z_i \Delta_i \quad \forall i = 1:\mathcal{N} \quad &: \pi^-_i \\
& \mu_i \geq \hat{r}_i - z_i \Delta_i  \quad \forall i = 1:\mathcal{N} \quad &: \pi^+_i \\
& z_i \geq 0 \quad \forall i = 1:\mathcal{N} \\
& z_i \leq 1 \quad \forall i = 1:\mathcal{N} \quad &: \theta_i \\
& \sum_{i}^{\mathcal{N}} z_i \leq \Gamma \quad : \lambda \\
\end{aligned}
```

However, the above equations cannot be directly incorporated in the upper-level problem since no of-the-shelf solver can solve the resulting bi-level ("MinMax") optimization problem. Moreover, our case becomes even harder given the variable multiplication of the upper-level variable (``w``) with the lower-level decision variable (``\mu``) in the objective function of the primal problem. The solution to this issue is to use of the following equivalent dual problem:

```math
\begin{aligned}
\max_{\lambda, \pi^-, \pi^+, \theta} \quad &  \sum_{i}^{\mathcal{N}} (\hat{r}_i (\pi^+_i - \pi^-_i) - \theta_i ) - \Gamma \lambda\\
s.t.  \quad & w_i = \pi^+_i - \pi^-_i  \quad \forall i = 1:\mathcal{N} \\
&  \Delta_i (\pi^+_i + \pi^-_i) - \theta_i \leq \lambda \quad \forall i = 1:\mathcal{N} \\
& \lambda \geq 0 , \; \pi^- \geq 0 , \; \pi^+ \geq 0 , \; \theta \geq 0 \\
\end{aligned}
```

Moreover, to avoid having a bi-level optimization problem, we replace the lower-level problem by its objective function expression and enforce the dual constraints in the upper-level problem, defining a lower bound for the optimal value (which will be exact if the upper-level problem requires). 

Finally, for instance, the resulting problem becomes:

```math
\begin{aligned}
\max_{w, \lambda, \pi^-, \pi^+, \theta} \quad & R \\
s.t. \quad & R = \sum_{i}^{\mathcal{N}} (\hat{r}_i (\pi^+_i \pi^-_i) - \theta_i ) - \Gamma \lambda \\
& w_i = \pi^+_i - \pi^-_i  \quad \forall i = 1:\mathcal{N} \\
&  \Delta_i (\pi^+_i + \pi^-_i) - \theta_i \leq \lambda \quad \forall i = 1:\mathcal{N} \\\\
& w ' \Sigma w  \leq V_0 * W_0 \\
& \lambda \geq 0 , \; \pi^- \geq 0 , \; \pi^+ \geq 0 , \; \theta \geq 0 \\
& w \in \mathcal{X} \\
\end{aligned}
```
#### Uncertainty Set Vizualization and Special Case (Soyster's Uncertainty Set)
In order to visualize Bertsimas's uncertainty set, it's useful to plot the case with only two assets. For instance, when the budget parameter is equal to one (``\Gamma = 1``) the resulting feasible region of the uncertaity set only allows one asset to be in its extreme value:

![](https://github.com/andrewrosemberg/PortfolioOpt/blob/master/docs/src/assets/set_bertsimas.png?raw=true)

On the other hand, when the budget parameter is equal to the number of assets (``\Gamma = 2``), the uncertainty set becomes similar to the one proposed by Soyster in [1], i.e. box uncertainty:

![](https://github.com/andrewrosemberg/PortfolioOpt/blob/master/docs/src/assets/set_soyster.png?raw=true)

#### Building Intuition with Efficient Frontiers
Understanding the impacts on the final optimal portfolio given the uncertainty set is not trivial, but some easy analysis can help. One useful analysis is to plot the Efficient Frontier (EF) (a.k.a Pareto Frontier) for our optimization problems, which shows what are the optimal porfolios given our objective and constraints.

If we plot the EF for the classic mean-variance problem (Markoviz) as well as the closest robust portfolios (e.g. Bertsimas and Soyster), we can see that by expanding our uncertainty set (and consequently restricting further our problem), we get sub-optimal solutions for the non-robust problem:

![](https://github.com/andrewrosemberg/PortfolioOpt/blob/master/docs/src/assets/pareto_markowitz.png?raw=true)

Its also useful to see the EF for the respective robust problems since they have their own efficient frontier:

![](https://github.com/andrewrosemberg/PortfolioOpt/blob/master/docs/src/assets/pareto_robust.png?raw=true)

Notwithstanding the importance of looking at EF for specific objectives, its more useful to see the possible returns the portfolio from a certain strategy might have if we allow the asset returns to vary inside the uncertainty sets. For instance, lets allow returns to vary inside Bertsimas' uncertainty set and see the consequences for Markowitz optimal portfolios and Bertsimas' optimal portfolios:

![](https://github.com/andrewrosemberg/PortfolioOpt/blob/master/docs/src/assets/uncertain_pareto_bertsimas.png?raw=true)

Now we can cleary see the trade-off the robust portfolio is providing: a lower avarage return for a smaller range of possible portfolio return that is contained in the range of the non-robust counterpart.

Moreover, if we allow returns to vary inside Soyster's uncertainty set, we get the following frontiers:

![](https://github.com/andrewrosemberg/PortfolioOpt/blob/master/docs/src/assets/uncertain_pareto_soyster.png?raw=true)

Once again we get a lower avarage return for a smaller range of possible portfolio return that is contained in the range of the less-robust counterpart.

PS.: Code in `examples/test_effi_robust.jl`.
### Comming Soon
TODO: Ben-Tal's uncertainty set ([`EllipticalSet`](@ref))

## References

[1] Soyster, A.L. Convex programming with set-inclusive constraints and applications to inexact linear
programming. Oper. Res. 1973, 21, 1154–1157.

[2] Ben-Tal, A. e Nemirovski, A. (1999). Robust solutions of uncertain linear programs. Operations research letters, 25(1):1–13. 

[3] Ben-Tal, A. e Nemirovski, A. (2000). Robust solutions of linear programming problems contaminated with uncertain data. Mathematical programming, 88(3):411–424. 

[4] Bertsimas, D. e Brown, D. B. (2009). Constructing uncertainty sets for robust linear optimization. Operations research, 57(6):1483–1495. 

[5] Bertsimas, D. e Pachamanova, D. (2008). Robust multiperiod portfolio management in the presence of transaction costs. Computers & Operations Research, 35(1):3–17. 

[6] Bertsimas, D. e Sim, M. (2004). The price of robustness. Operations research, 52(1):35–53. 

[7] Bertsimas, D. e Sim, M. (2006). Tractable approximations to robust conic optimization problems. Mathematical programming, 107(1-2):5–36. 

[8] Fabozzi, F. J., Huang, D., e Zhou, G. (2010). Robust portfolios: contributions from operations research and finance. Annals of Operations Research, 176(1):191–220. 

[9] Fabozzi, F. J., Kolm, P. N., Pachamanova, D. A., e Focardi, S. M. (2007). Robust portfolio optimization. Journal of Portfolio Management, 33(3):40. 

[10] Kim, J. H., Kim, W. C., e Fabozzi, F. J. (2014). Recent developments in robust portfolios with a worst-case approach. Journal of Optimization Theory and Applications, 161(1):103–121.

[11] Natarajan, K., Pachamanova, D., e Sim, M. (2009). Constructing risk measures from uncertainty sets. Operations research, 57(5):1129–1141.

[12] Fernandes, B., Street, A., ValladA˜ £o, D., e Fernandes, C. (2016). An adaptive robust portfolio
optimization model with loss constraints based on data-driven polyhedral uncertainty sets. European Journal of Operational Research, 255(3):961 – 970. ISSN 0377-2217. [URL](www.sciencedirect.com/science/article/pii/S0377221716303757).

[13] Futures Contracts Portfolio Selection via Robust Data Driven Optimization publication date Aug 9, 2018  publication descriptionL SBPO, 2018, Rio de Janeiro. Anais do L SBPO, 2018. v. 1. [URL](https://proceedings.science/sbpo/papers/selecao-de-carteira-de-contratos-futuros-via-otimizacao-robusta-direcionado-por-dados).

