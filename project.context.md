# Project Context: Portfolio Construction Shootout

## Purpose
This is a portfolio project for a quant position application at a university student managed fund. It needs to be finished in about one week (roughly 20 hours total), look professional on GitHub, and be something the author can confidently explain in an interview.

The author is a student using Claude Code for most of the implementation. **Code must be clear, well commented, and easy for a finance student to follow.** Favour readability over cleverness.

## Research Question
> Does portfolio optimisation actually beat a simple equal-weight portfolio out of sample, once realistic constraints and transaction costs are included, and if not, why not?

Background: DeMiguel, Garlappi & Uppal (2009), "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?", *Review of Financial Studies*. They found naive 1/N diversification is very hard to beat because optimisers amplify estimation error, especially in expected returns.

## Strategies Compared
1. **Equal weight (1/N)**: the benchmark
2. **Minimum variance**: uses only the covariance matrix
3. **Max Sharpe (mean-variance)**: uses the sample mean and covariance
4. **Risk parity**: each asset contributes equally to portfolio risk
5. **Minimum variance with Ledoit-Wolf shrinkage**: tests whether better covariance estimates help

SPY is included as an external benchmark only. It is not part of the investable universe.

## Expected Findings (hypotheses to test, not assume)
- Max Sharpe will have unstable weights, high turnover, and weak out-of-sample performance.
- Minimum variance and risk parity will deliver better risk-adjusted returns than max Sharpe.
- Equal weight will be hard to beat after costs.
- Shrinkage will stabilise minimum variance weights and reduce turnover.

Report whatever the data actually shows, even if it contradicts these.

## Deliverables
1. Working Python code that runs end to end with a single command (`python main.py`).
2. Results table: annualised return, volatility, Sharpe ratio, max drawdown, average turnover (all after costs).
3. Charts: cumulative returns, drawdowns, weights over time for each strategy.
4. Robustness checks: different estimation windows (1, 3, 5 years) and transaction costs (0, 10, 25 bps).
5. A strong `README.md` covering the question, data, method, key results with charts, findings, limitations, and possible extensions.

## Timeline
| Days | Goal |
|------|------|
| 1–2 | Data pipeline and core backtest working for all strategies |
| 3–4 | Verify results, add shrinkage strategy, polish charts |
| 5 | Robustness checks |
| 6–7 | README, final cleanup, CV bullet |

## Scope Rules
- **In scope:** everything above.
- **Out of scope unless the core is completely finished:** web apps, dashboards, machine learning, Black-Litterman, options, leverage, short selling.
- Keep it simple, correct, and well explained. Correctness and clarity matter more than features.

## Interview Questions the Project Must Help Answer
- Why did mean-variance perform the way it did?
- What does risk parity equalise, and why does that help?
- How was look-ahead bias avoided?
- What does Ledoit-Wolf shrinkage do?
- What are the limitations, and what would you do with more time?
