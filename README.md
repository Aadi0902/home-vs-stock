# 🏠 vs 📈 — Home vs. Stock Market Decision Dashboard

An interactive Streamlit dashboard that answers two questions at once:

1. **Is buying a house even worth it** versus staying in your current rental and
   investing the difference?
2. **If you do have capital, where does it grow more** — a home (with roommates
   paying rent) or the S&P 500?

It does this with a fair, apples-to-apples model, Monte Carlo **risk /
expected-value** simulation, and a built-in **optimizer**.

## Quick start

```bash
pip install -r requirements.txt
streamlit run home_vs_stock.py
```

Your browser opens the dashboard. Tune anything in the left sidebar; every chart
and number updates live.

## How the comparison is kept fair

Both strategies start from the **same up-front capital** and the **same monthly
disposable income**. Each month your take-home pay minus living expenses is the
cash available for housing + investing; both scenarios draw from that same pool,
so the comparison is fair. Whatever isn't spent on housing is the investable
**surplus**:

- **RENT** invests its whole surplus in the S&P 500.
- **BUY** splits its surplus between **extra mortgage principal** (a guaranteed
  return equal to your mortgage rate) and the **S&P 500** — a tunable dial. Extra
  principal shortens the loan; once it's paid off the freed-up payment flows
  straight into investments.

The two headline strategies remain:

- **BUY:** buy the home, live in it, rent spare rooms to roommates, pay the full
  cost of ownership (mortgage P&I, property tax, insurance, PMI, maintenance,
  HOA).
- **RENT:** keep paying your **current rent**, invest the down-payment + closing
  costs in the S&P 500 at day one, and each month invest whatever the cheaper
  strategy saves.

Because the monthly budget is equalized (the cheaper side invests the surplus),
comparing terminal net worth is fair. All outcomes are reported **after** selling
costs, capital-gains taxes, and the home-sale exclusion.

## Auto-fill from your ZIP code

Type a 5-digit U.S. ZIP into the sidebar's **📍 Auto-fill from ZIP code** box and
click **Apply local defaults**. The app resolves the ZIP to its state and fills
in state-level **property tax, homeowners insurance, home appreciation, and
median home price** — then you can fine-tune anything. Sources: Tax Foundation
(property tax), Zillow/Census (median price), NerdWallet (insurance);
appreciation is a regional estimate. These are state-level figures keyed off the
ZIP prefix, not exact per-ZIP data — override with local comps. See
`market_data.py`; a paid API (ATTOM/RentCast/Zillow) can be slotted into
`lookup_zip()` for true per-ZIP precision.

## What's tunable

ZIP code (auto-fill), home price, down payment %, buy/sell closing costs, holding
horizon, mortgage rate & term, property tax, insurance, PMI, maintenance, HOA,
roommate rent + **how many months you'll keep roommates** + growth + vacancy +
tax, **your current rent** + growth, **take-home income + living-expense % +
income growth**, **% of surplus paid to extra mortgage principal vs. invested**,
S&P 500 expected return & volatility, home appreciation & volatility, stock–home
correlation, inflation, capital-gains taxes, the §121 home-sale exclusion, and
the mortgage-interest / property-tax deduction (with SALT cap).

### Pay-down vs. invest

The optimizer tab includes a dedicated **"pay down the mortgage vs. invest"**
sweep. With the *expected-wealth* objective it usually favors 0% extra principal
(stocks' higher expected return wins); switch to the *risk-adjusted* objective
and raise the risk-aversion λ, and paying down the guaranteed-return mortgage
starts to win because it removes risk. Example (15-yr horizon): 0% extra
principal → ~$2.04M expected but $648k σ; 100% → ~$1.77M expected but only $308k
σ and a *higher* worst-case outcome.

## The five tabs

| Tab | What it shows |
|---|---|
| 📈 **Net worth over time** | Expected-path net worth for both strategies, the break-even year, and home equity build-up. |
| 🎲 **Risk / distribution** | Monte Carlo outcome histograms, downside/upside percentiles, a Sharpe-like ratio, and a return-vs-risk plot. |
| 🎯 **Optimizer** | Grid-search heatmap over down-payment % and horizon; maximizes expected or risk-adjusted (mean − λ·σ) wealth. |
| 🌪 **Sensitivity** | Tornado chart ranking which assumptions swing the answer most. |
| 📚 **Data & assumptions** | Every default value with its source and the method/caveats. |

## Default data sources (mid-2026, U.S.)

- **S&P 500:** ~10% mean return, ~18% volatility (Slickcharts, Macrotrends; long-run large-cap std ≈ 19.8%).
- **Home appreciation:** ~4.5% nominal, ~7% volatility (S&P/Case-Shiller via FRED).
- **Mortgage rate:** ~6.5% (Freddie Mac 30-yr PMMS, week of Jul 2 2026 ≈ 6.43%).
- **Property tax:** ~1.1%/yr (Tax Foundation / Census). **Insurance:** ~0.6%/yr (NerdWallet/Forbes).
- **Maintenance:** ~1%/yr (1% rule). **PMI:** ~0.7%/yr when <20% down.
- **Taxes:** 15% long-term capital gains; §121 $250k/$500k home-sale exclusion; $10k SALT cap.
- **Inflation:** 3%/yr.

## Caveats

This is an **educational planning tool, not financial advice.** Returns are
modeled as normal & i.i.d. by year (no fat tails or mean reversion), the mortgage
deduction is simplified (can slightly overstate the owner's benefit — uncheck it
to be conservative), and rental income uses a single effective tax rate. Verify
current rates for your own situation before acting.
