# Robust Portfolio Optimization
**Acknowledgements**: Robust Formulations based on and inspired by [Davi M. Valladão](http://www.ind.puc-rio.br/en/equipe/davi-michel-valladao/)'s lectures on "Capital Market".

## Motivation
Portfolio Optimization (PO) formulations were developed to adapt to a variety of settings for decisions under uncertainty. These formulations depend on the available information of the uncertain data and the risk aversion of the decision-maker.

The available information for many market data (e.g. stock prices, future prices, etc) have non-stationary profiles that are hard to make statistical inference on. Thus, it can be hard to find good probability distributions needed for stochastic optimization. 

One alternative approach is Robust Optimization.

## Background
Robust Optimization (RO) problems belong to the class of optimization under uncertainty problems where some problem data is uncertain (either because the decision is taken before the realization of the random event or because its observation is not available). Usual application cases for RO are when there isn’t sufficient information to derive probability distributions, but this isn’t strictly necessary. RO focuses on guaranteeing solution feasibility for any possible value of the uncertain data inside a defined Uncertainty Set. In the case where the uncertainty impacts the objective function, it guarantees optimality for the works case scenario considered in the Uncertainty Set.

Many uncertainty sets have been proposed to accommodate different levels of conservatism and data structures ([1]-[6]). A comparison of uncertainty sets to usual risk measures used in finance was made in [3] and [10].

A collection of recent contributions to robust portfolio strategies was made in [7 - 10]. Data-driven approaches to robust PO also gained interest in recent years and can be found in [11] (for a portfolio of stocks) and [12] (for a portfolio of future contracts). The results in those studies indicate promising alternatives for the integration between uncertain data and PO.

## Problem Definition
Simple versions of the Mean-Variance PO problem with robust uncertainty around the estimated mean returns are implemented by the following functions:

```@docs
po_min_variance_limit_return!
```

```@docs
po_max_return_limit_variance!
```

## Example case Bertsimas

## References
[1] Ben-Tal, A. e Nemirovski, A. (1999). Robust solutions of uncertain linear programs. Operations research letters, 25(1):1–13. 

[2] Ben-Tal, A. e Nemirovski, A. (2000). Robust solutions of linear programming problems contaminated with uncertain data. Mathematical programming, 88(3):411–424. 

[3] Bertsimas, D. e Brown, D. B. (2009). Constructing uncertainty sets for robust linear optimization. Operations research, 57(6):1483–1495. 

[4] Bertsimas, D. e Pachamanova, D. (2008). Robust multiperiod portfolio management in the presence of transaction costs. Computers & Operations Research, 35(1):3–17. 

[5] Bertsimas, D. e Sim, M. (2004). The price of robustness. Operations research, 52(1):35–53. 

[6] Bertsimas, D. e Sim, M. (2006). Tractable approximations to robust conic optimization problems. Mathematical programming, 107(1-2):5–36. 

[7] Fabozzi, F. J., Huang, D., e Zhou, G. (2010). Robust portfolios: contributions from operations research and finance. Annals of Operations Research, 176(1):191–220. 

[8] Fabozzi, F. J., Kolm, P. N., Pachamanova, D. A., e Focardi, S. M. (2007). Robust portfolio optimization. Journal of Portfolio Management, 33(3):40. 

[9] Kim, J. H., Kim, W. C., e Fabozzi, F. J. (2014). Recent developments in robust portfolios with a worst-case approach. Journal of Optimization Theory and Applications, 161(1):103–121.

[10] Natarajan, K., Pachamanova, D., e Sim, M. (2009). Constructing risk measures from uncertainty sets. Operations research, 57(5):1129–1141.

[11] Fernandes, B., Street, A., ValladA˜ £o, D., e Fernandes, C. (2016). An adaptive robust portfolio
optimization model with loss constraints based on data-driven polyhedral uncertainty sets. European Journal of Operational Research, 255(3):961 – 970. ISSN 0377-2217. [URL](www.sciencedirect.com/science/article/pii/S0377221716303757).

[12] Futures Contracts Portfolio Selection via Robust Data Driven Optimization publication date Aug 9, 2018  publication descriptionL SBPO, 2018, Rio de Janeiro. Anais do L SBPO, 2018. v. 1. [URL](https://proceedings.science/sbpo/papers/selecao-de-carteira-de-contratos-futuros-via-otimizacao-robusta-direcionado-por-dados).

