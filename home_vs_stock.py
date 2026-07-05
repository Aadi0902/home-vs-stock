"""
Home vs. Stock Market — Investment Decision Dashboard
=====================================================

An interactive Streamlit dashboard that compares two ways of deploying the same
capital and the same monthly housing budget:

  Scenario A  "BUY"   — Buy a home, live in it, rent spare rooms to roommates,
                        pay the full cost of ownership (PITI + maintenance).
  Scenario B  "RENT"  — Rent a place to live, invest the down-payment + closing
                        costs in an S&P 500 index fund, and invest any monthly
                        cash-flow difference between the two scenarios.

Because the monthly budget is equalized (whichever scenario is cheaper in a
given month invests the surplus), this is a fair, apples-to-apples comparison of
terminal net worth.

It includes:
  * Every factor tunable from the sidebar, with realistic, data-based defaults.
  * A Monte Carlo engine that treats stock returns and home appreciation as
    uncertain (mean + standard deviation + correlation) so you get an EXPECTED
    VALUE and a full risk distribution, not just a point estimate.
  * Charts: net-worth-over-time, outcome distributions, and an optimization
    heatmap.
  * An OPTIMIZER that searches down-payment % and holding horizon to maximize
    either expected wealth or risk-adjusted wealth (mean - λ·σ).

Run with:   streamlit run home_vs_stock.py

Data sources for the default values are listed in the "Data sources &
assumptions" expander in the app and in README.md.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import market_data as md

st.set_page_config(page_title="Home vs. Stock Market", page_icon="🏠", layout="wide")


def _snap(value, step):
    """Round a value to the nearest slider step so session-state seeds are valid."""
    return round(round(value / step) * step, 4)


# Fields that a ZIP lookup can auto-fill. Seed session_state so the keyed widgets
# below have a starting value (and so the ZIP button can overwrite them).
_ZIP_DEFAULTS = {"home_price": 500_000, "prop_tax_pct": 1.10,
                 "insurance_pct": 0.60, "home_mean_pct": 4.50}
for _k, _v in _ZIP_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# ---------------------------------------------------------------------------
# CORE SIMULATION ENGINE
# ---------------------------------------------------------------------------


def mortgage_payment(loan0, annual_rate, term_years):
    """Scheduled monthly principal & interest for a fixed, fully-amortizing loan.
    Extra principal payments shorten the term but do NOT change this payment."""
    if loan0 <= 0:
        return 0.0
    mr = annual_rate / 12.0
    n = int(round(term_years * 12))
    if mr > 0:
        return loan0 * mr * (1 + mr) ** n / ((1 + mr) ** n - 1)
    return loan0 / n


def run_sim(mf_stock, mf_home, p, record_path=False):
    """Vectorized month-by-month simulation across n_sims parallel scenarios.

    mf_stock, mf_home : (n_sims, months) arrays of MONTHLY growth factors.

    Cash model (same disposable income for both scenarios, so it's fair):
        available = after-tax income × (1 − expenditure%)   [grows w/ income]
    Each month both scenarios pay their housing cost out of `available`; whatever
    is left over is the investable SURPLUS.
      * RENT: the whole surplus goes into the S&P 500.
      * BUY:  a chosen fraction of the surplus is thrown at EXTRA mortgage
              principal (a guaranteed return = the mortgage rate); the rest goes
              into the S&P 500. Extra principal shortens the loan; once it's paid
              off the freed-up P&I becomes additional surplus automatically.
    """
    n_sims, months = mf_stock.shape
    P0 = p["home_price"]
    loan0 = P0 * (1.0 - p["down_pct"])

    # Capital deployed up front to buy = down payment + buying closing costs.
    # In the RENT scenario this exact amount is invested in stocks at t=0.
    C0 = P0 * p["down_pct"] + P0 * p["closing_buy"]

    mr = p["mortgage_rate"] / 12.0
    term_m = int(round(p["loan_term"] * 12))
    pmt = mortgage_payment(loan0, p["mortgage_rate"], p["loan_term"])
    salt_cap_month = p["salt_cap"] / 12.0
    extra_pct = p["extra_principal_pct"]

    home_value = np.full(n_sims, P0, dtype=float)
    balance = np.full(n_sims, loan0, dtype=float)   # per-sim: extra payments vary it
    owner_side = np.zeros(n_sims)      # side stock account in BUY scenario
    owner_basis = np.zeros(n_sims)     # cost basis (for cap-gains tax)
    renter_inv = np.full(n_sims, C0)   # stock account in RENT scenario
    renter_basis = np.full(n_sims, C0)

    if record_path:
        owner_nw = np.zeros((n_sims, months))
        renter_nw = np.zeros((n_sims, months))
        home_path = np.zeros((n_sims, months))
        balance_path = np.zeros((n_sims, months))

    for m in range(months):
        yr = m // 12
        home_value *= mf_home[:, m]

        # --- Regular mortgage payment (per-sim, since balance differs) ---
        active = (balance > 1e-6) & (m < term_m)
        interest = np.where(active, balance * mr, 0.0)
        reg_principal = np.where(active, np.minimum(pmt - interest, balance), 0.0)
        pi_pay = interest + reg_principal
        balance = balance - reg_principal

        # --- Cost of ownership this month ---
        prop_tax = home_value * p["prop_tax"] / 12.0
        insurance = home_value * p["insurance"] / 12.0
        maintenance = home_value * p["maintenance"] / 12.0
        hoa = p["hoa"]
        ltv = np.where(home_value > 0, balance / home_value, 0.0)
        pmi = np.where(ltv > 0.80, balance * p["pmi"] / 12.0, 0.0)

        # --- Roommate income (taxable); stops after roommate_months ---
        if m < p["roommate_months"]:
            rent_income = (
                p["roommate_rent"] * (1 + p["rent_growth"]) ** yr * (1 - p["vacancy"])
            )
        else:
            rent_income = 0.0
        rent_after_tax = rent_income * (1 - p["rent_tax"])

        # --- Mortgage-interest + property-tax deduction (optional) ---
        if p["deduct"]:
            deductible = interest + np.minimum(prop_tax, salt_cap_month)
            tax_shield = deductible * p["marginal_rate"]
        else:
            tax_shield = 0.0

        owner_housing = (
            pi_pay + prop_tax + insurance + maintenance + hoa + pmi
            - rent_after_tax - tax_shield
        )
        renter_housing = p["your_rent"] * (1 + p["your_rent_growth"]) ** yr

        # --- Disposable income available for housing + investing (same both) ---
        available = (p["after_tax_income"] * (1 - p["expenditure_pct"])
                     * (1 + p["income_growth"]) ** yr)
        owner_surplus = np.maximum(available - owner_housing, 0.0)
        renter_surplus = np.maximum(available - renter_housing, 0.0)

        # --- BUY: split surplus between extra principal and the S&P 500 ---
        extra_principal = owner_surplus * extra_pct
        extra_used = np.minimum(extra_principal, balance)   # can't overpay the loan
        balance = balance - extra_used
        to_stocks_owner = owner_surplus - extra_used        # leftover auto-flows to stocks

        owner_side = owner_side * mf_stock[:, m] + to_stocks_owner
        owner_basis = owner_basis + to_stocks_owner
        renter_inv = renter_inv * mf_stock[:, m] + renter_surplus
        renter_basis = renter_basis + renter_surplus

        if record_path:
            eq = home_value * (1 - p["closing_sell"]) - balance
            hg = np.maximum(0.0, (home_value - P0) - p["home_exclusion"])
            htax = hg * p["home_cg"]
            oside_net = owner_side - np.maximum(0.0, owner_side - owner_basis) * p["stock_cg"]
            owner_nw[:, m] = eq - htax + oside_net
            renter_nw[:, m] = renter_inv - np.maximum(0.0, renter_inv - renter_basis) * p["stock_cg"]
            home_path[:, m] = home_value
            balance_path[:, m] = balance

    # --- Terminal wealth (after selling costs, cap-gains taxes) ---
    owner_equity = home_value * (1 - p["closing_sell"]) - balance
    home_gain_taxable = np.maximum(0.0, (home_value - P0) - p["home_exclusion"])
    home_tax = home_gain_taxable * p["home_cg"]
    owner_side_net = owner_side - np.maximum(0.0, owner_side - owner_basis) * p["stock_cg"]
    owner_terminal = owner_equity - home_tax + owner_side_net

    renter_terminal = renter_inv - np.maximum(0.0, renter_inv - renter_basis) * p["stock_cg"]

    # Inflation-adjust to today's dollars if requested.
    if p["real_terms"]:
        deflator = (1 + p["inflation"]) ** p["horizon_years"]
        owner_terminal = owner_terminal / deflator
        renter_terminal = renter_terminal / deflator

    out = {
        "owner_terminal": owner_terminal,
        "renter_terminal": renter_terminal,
        "C0": C0,
        "loan0": loan0,
        "monthly_pi": pmt,
    }
    if record_path:
        if p["real_terms"]:
            defl = (1 + p["inflation"]) ** (np.arange(1, months + 1) / 12.0)
            owner_nw = owner_nw / defl
            renter_nw = renter_nw / defl
        out.update(
            owner_nw=owner_nw[0],
            renter_nw=renter_nw[0],
            home_path=home_path[0],
            balance_path=balance_path[0],
        )
    return out


def monthly_factors(annual_returns):
    """Convert an (n_sims, n_years) array of annual returns into an
    (n_sims, months) array of monthly growth factors. Annual returns are
    clipped at -95% so the monthly root stays real and sane."""
    n_sims, n_years = annual_returns.shape
    clipped = np.clip(annual_returns, -0.95, None)
    monthly = (1 + clipped) ** (1 / 12.0)
    return np.repeat(monthly, 12, axis=1)


@st.cache_data(show_spinner=False)
def monte_carlo(p, n_sims=3000, seed=42):
    """Draw correlated annual stock/home returns and run the sim across n_sims."""
    months = int(p["horizon_years"] * 12)
    n_years = int(p["horizon_years"])
    rng = np.random.default_rng(seed)

    means = [p["stock_mean"], p["home_mean"]]
    ss, sh, rho = p["stock_std"], p["home_std"], p["corr"]
    cov = [[ss * ss, rho * ss * sh], [rho * ss * sh, sh * sh]]
    draws = rng.multivariate_normal(means, cov, size=(n_sims, n_years))  # (sims,yrs,2)

    mf_stock = monthly_factors(draws[:, :, 0])[:, :months]
    mf_home = monthly_factors(draws[:, :, 1])[:, :months]
    return run_sim(mf_stock, mf_home, p, record_path=False)


def deterministic_path(p):
    """Single expected-return path (mean returns, no volatility) with the full
    monthly time series recorded for charting."""
    months = int(p["horizon_years"] * 12)
    n_years = int(p["horizon_years"])
    stock = np.full((1, n_years), p["stock_mean"])
    home = np.full((1, n_years), p["home_mean"])
    mf_stock = monthly_factors(stock)[:, :months]
    mf_home = monthly_factors(home)[:, :months]
    return run_sim(mf_stock, mf_home, p, record_path=True)


# ---------------------------------------------------------------------------
# SIDEBAR — every factor is tunable here
# ---------------------------------------------------------------------------

st.sidebar.title("🏠 vs 📈  Inputs")
st.sidebar.caption("Defaults are realistic U.S. figures (mid-2026). Tune anything.")

with st.sidebar.expander("📍 Auto-fill from ZIP code", expanded=True):
    zipcode = st.text_input("ZIP code", value="", max_chars=5,
                            placeholder="e.g. 94103")
    if st.button("Apply local defaults", width='stretch'):
        info = md.lookup_zip(zipcode)
        if info:
            st.session_state["prop_tax_pct"] = _snap(info["property_tax"] * 100, 0.05)
            st.session_state["insurance_pct"] = _snap(info["insurance"] * 100, 0.05)
            st.session_state["home_mean_pct"] = _snap(info["home_appreciation"] * 100, 0.25)
            st.session_state["home_price"] = int(info["median_home_price"])
            st.success(
                f"Loaded **{info['state_name']}** averages: "
                f"price \\${info['median_home_price']:,}, "
                f"property tax {info['property_tax']*100:.2f}%, "
                f"insurance {info['insurance']*100:.2f}%, "
                f"appreciation {info['home_appreciation']*100:.1f}%. "
                "Adjust any field below.")
        else:
            st.error("ZIP not recognized — enter a valid 5-digit U.S. ZIP, "
                     "or set values manually below.")
    st.caption("State-level figures from Tax Foundation (property tax), "
               "Zillow/Census (price), NerdWallet (insurance); appreciation is a "
               "regional estimate — override with local comps.")

with st.sidebar.expander("① The Home & Purchase", expanded=True):
    home_price = st.number_input("Home price ($)", 50_000, 5_000_000, step=10_000,
                                 key="home_price")
    down_pct = st.slider("Down payment (%)", 0.0, 100.0, 20.0, 1.0) / 100
    closing_buy = st.slider("Buying closing costs (% of price)", 0.0, 6.0, 3.0, 0.25) / 100
    closing_sell = st.slider("Selling costs when you sell (% of price)", 0.0, 12.0, 8.0, 0.5) / 100
    horizon_years = st.slider("Holding horizon (years)", 1, 30, 10, 1)

with st.sidebar.expander("② Mortgage (PITI)", expanded=True):
    mortgage_rate = st.slider("Mortgage interest rate (%)", 0.0, 12.0, 6.5, 0.05) / 100
    loan_term = st.selectbox("Loan term (years)", [15, 20, 30], index=2)
    prop_tax = st.slider("Property tax (% of value / yr)", 0.0, 3.0, step=0.05,
                         key="prop_tax_pct") / 100
    insurance = st.slider("Homeowners insurance (% of value / yr)", 0.0, 2.0, step=0.05,
                          key="insurance_pct") / 100
    pmi = st.slider("PMI (% of loan / yr, if <20% down)", 0.0, 2.0, 0.7, 0.05) / 100

with st.sidebar.expander("③ Ownership operating costs", expanded=False):
    maintenance = st.slider("Maintenance & repairs (% of value / yr)", 0.0, 4.0, 1.0, 0.1) / 100
    hoa = st.number_input("HOA / condo fees ($/mo)", 0, 3000, 0, 25)

with st.sidebar.expander("④ Roommate rental income", expanded=True):
    roommate_rent = st.number_input("Roommate rent collected ($/mo)", 0, 20_000, 1_200, 50)
    roommate_months = st.slider("Months you'll have roommates", 0, horizon_years * 12,
                                min(60, horizon_years * 12), 1,
                                help="Roommate income stops after this many months "
                                     "(e.g. once you want the place to yourself).")
    rent_growth = st.slider("Rent income growth (%/yr)", 0.0, 8.0, 3.0, 0.25) / 100
    vacancy = st.slider("Vacancy rate (%)", 0.0, 30.0, 5.0, 1.0) / 100
    rent_tax = st.slider("Effective tax on rental income (%)", 0.0, 50.0, 15.0, 1.0) / 100
    st.caption(f"You'll collect roommate rent for **{roommate_months} of "
               f"{horizon_years * 12} months** ({roommate_months / 12:.1f} yrs). "
               "Effective tax rate is below your marginal rate because landlords "
               "deduct expenses & depreciation.")

with st.sidebar.expander("⑤ Your CURRENT rent (the 'don't buy' case)", expanded=True):
    your_rent = st.number_input("Your current rent ($/mo)", 0, 20_000, 2_200, 50,
                                help="What you pay for housing today and would keep "
                                     "paying if you DON'T buy. This is the true "
                                     "alternative — it decides if buying is worth it.")
    your_rent_growth = st.slider("Expected rent growth (%/yr)", 0.0, 8.0, 3.0, 0.25) / 100
    st.caption("If buying can't beat simply staying in this rental and investing "
               "the difference, buying isn't worth it for you.")

with st.sidebar.expander("⑥ Your income & surplus allocation", expanded=True):
    after_tax_income = st.number_input("Take-home (after-tax) income ($/mo)",
                                       0, 100_000, 9_000, 250)
    expenditure_pct = st.slider("Living expenses (% of income, non-housing)",
                                0.0, 90.0, 35.0, 1.0) / 100
    income_growth = st.slider("Income growth (%/yr)", 0.0, 10.0, 3.0, 0.25) / 100
    _avail = after_tax_income * (1 - expenditure_pct)
    st.caption(f"After expenses, **\\${_avail:,.0f}/mo** is available for housing + "
               "investing. Both scenarios draw from the same amount, so the "
               "comparison stays fair.")
    extra_principal_pct = st.slider(
        "Of the BUY surplus, % thrown at EXTRA mortgage principal", 0.0, 100.0, 0.0, 5.0,
        help="0% = invest the whole surplus in the S&P 500. 100% = pay the "
             "mortgage down as fast as possible, then invest once it's gone. "
             "Extra principal earns a guaranteed return equal to your mortgage "
             "rate; stocks have a higher expected but risky return.") / 100
    st.caption("The rest of the BUY surplus goes into the S&P 500. This is the "
               "'pay down the house vs. invest' dial. The optimizer tab finds "
               "the sweet spot.")

with st.sidebar.expander("⑦ Market assumptions & RISK", expanded=True):
    stock_mean = st.slider("S&P 500 expected return (%/yr)", 0.0, 15.0, 10.0, 0.25) / 100
    stock_std = st.slider("S&P 500 volatility σ (%/yr)", 0.0, 40.0, 18.0, 0.5) / 100
    home_mean = st.slider("Home appreciation (%/yr)", -2.0, 12.0, step=0.25,
                          key="home_mean_pct") / 100
    home_std = st.slider("Home appreciation volatility σ (%/yr)", 0.0, 25.0, 7.0, 0.5) / 100
    corr = st.slider("Stock–home return correlation", -1.0, 1.0, 0.2, 0.05)
    inflation = st.slider("Inflation (%/yr)", 0.0, 8.0, 3.0, 0.25) / 100

with st.sidebar.expander("⑧ Taxes on exit", expanded=False):
    stock_cg = st.slider("Capital-gains tax on stocks (%)", 0.0, 40.0, 15.0, 1.0) / 100
    home_cg = st.slider("Capital-gains tax on home (%)", 0.0, 40.0, 15.0, 1.0) / 100
    home_exclusion = st.number_input("Home-sale gain exclusion ($)", 0, 500_000, 250_000, 50_000)
    deduct = st.checkbox("Deduct mortgage interest + property tax", value=True)
    marginal_rate = st.slider("Marginal income-tax rate (for deduction) (%)", 0.0, 50.0, 24.0, 1.0) / 100
    salt_cap = st.number_input("SALT deduction cap ($/yr)", 0, 100_000, 10_000, 1_000)

with st.sidebar.expander("⑨ Display", expanded=False):
    real_terms = st.checkbox("Show in today's dollars (inflation-adjusted)", value=False)
    n_sims = st.select_slider("Monte Carlo simulations", [1000, 3000, 5000, 10000], value=3000)

params = dict(
    home_price=home_price, down_pct=down_pct, closing_buy=closing_buy,
    closing_sell=closing_sell, horizon_years=horizon_years, mortgage_rate=mortgage_rate,
    loan_term=loan_term, prop_tax=prop_tax, insurance=insurance, pmi=pmi,
    maintenance=maintenance, hoa=hoa, roommate_rent=roommate_rent,
    roommate_months=roommate_months, rent_growth=rent_growth,
    vacancy=vacancy, rent_tax=rent_tax, your_rent=your_rent, your_rent_growth=your_rent_growth,
    after_tax_income=after_tax_income, expenditure_pct=expenditure_pct,
    income_growth=income_growth, extra_principal_pct=extra_principal_pct,
    stock_mean=stock_mean, stock_std=stock_std, home_mean=home_mean, home_std=home_std,
    corr=corr, inflation=inflation, stock_cg=stock_cg, home_cg=home_cg,
    home_exclusion=home_exclusion, deduct=deduct, marginal_rate=marginal_rate,
    salt_cap=salt_cap, real_terms=real_terms,
)

# ---------------------------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------------------------

st.title("🏠 Home vs. 📈 Stock Market — Which Wins?")
st.markdown(
    "Same up-front capital, same monthly housing budget, two strategies. "
    "**Buy** a home & rent rooms, or **rent** & invest the difference in the S&P 500. "
    "All outcomes are after transaction costs and taxes."
)

det = deterministic_path(params)
mc = monte_carlo(params, n_sims=n_sims)

owner_mc = mc["owner_terminal"]
renter_mc = mc["renter_terminal"]
unit = "today's $" if real_terms else "nominal $"


def fmt(x):
    """Plain money string for st.metric values / dataframes (not markdown)."""
    return f"${x:,.0f}"


def mfmt(x):
    """Money string safe for MARKDOWN: escapes '$' so Streamlit doesn't treat a
    pair of dollar signs as a LaTeX math expression."""
    return f"\\${x:,.0f}"


# ---- Headline metrics (expected values) ----
st.subheader(f"Expected outcome after {horizon_years} years  ({unit})")
c1, c2, c3, c4 = st.columns(4)
o_mean, r_mean = owner_mc.mean(), renter_mc.mean()
c1.metric("🏠 BUY — expected net worth", fmt(o_mean))
c2.metric("📈 RENT+INVEST — expected net worth", fmt(r_mean))
edge = o_mean - r_mean
c3.metric("Buy advantage (expected)", fmt(edge),
          delta="Buying wins" if edge > 0 else "Renting wins")
p_buy_wins = float((owner_mc > renter_mc).mean())
c4.metric("P(buying beats renting)", f"{p_buy_wins*100:.0f}%")

st.caption(
    f"Up-front capital deployed either way: **{mfmt(mc['C0'])}** "
    f"(down payment + buying costs). Mortgage: **{mfmt(mc['loan0'])}** at "
    f"{mortgage_rate*100:.2f}% → **{mfmt(mc['monthly_pi'])}/mo** principal & interest."
)

# ---- Plain-English verdict: is buying even worth it vs. staying in your rental? ----
margin = edge / max(abs(r_mean), 1)
if p_buy_wins >= 0.60 and edge > 0:
    st.success(
        f"✅ **Buying looks worth it.** Vs. staying in your **{mfmt(your_rent)}/mo** rental "
        f"and investing the difference, buying is expected to leave you **{mfmt(edge)}** "
        f"richer after {horizon_years} yrs and wins in **{p_buy_wins*100:.0f}%** of scenarios."
    )
elif p_buy_wins <= 0.40 or edge < 0:
    st.error(
        f"❌ **Buying may not be worth it.** Staying in your **{mfmt(your_rent)}/mo** rental "
        f"and investing the difference is expected to leave you **{mfmt(-edge)}** richer. "
        f"Buying only wins in **{p_buy_wins*100:.0f}%** of scenarios — try a longer horizon, "
        "more roommate income, or a lower price."
    )
else:
    st.warning(
        f"⚖️ **It's roughly a toss-up.** Buying wins in **{p_buy_wins*100:.0f}%** of scenarios "
        f"with an expected edge of just **{mfmt(edge)}**. The non-financial factors "
        "(stability, flexibility, effort) probably decide it."
    )

tab_time, tab_dist, tab_opt, tab_sens, tab_data = st.tabs(
    ["📈 Net worth over time", "🎲 Risk / distribution", "🎯 Optimizer",
     "🌪 Sensitivity", "📚 Data & assumptions"]
)

# ===========================================================================
# TAB 1 — Net worth over time (expected path)
# ===========================================================================
with tab_time:
    st.markdown("#### Liquidation net worth over time (expected / mean-return path)")
    months = np.arange(1, len(det["owner_nw"]) + 1)
    yrs = months / 12.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yrs, y=det["owner_nw"], name="🏠 Buy (net worth if sold)",
                             line=dict(width=3, color="#2563eb")))
    fig.add_trace(go.Scatter(x=yrs, y=det["renter_nw"], name="📈 Rent + invest",
                             line=dict(width=3, color="#16a34a")))
    fig.update_layout(height=430, xaxis_title="Year", yaxis_title=f"Net worth ({unit})",
                      hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')

    # crossover
    diff = det["owner_nw"] - det["renter_nw"]
    cross = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if len(cross):
        st.info(f"⚖️ Break-even (crossover) around **year {yrs[cross[0]]:.1f}** — "
                "before this, the other strategy is ahead.")
    else:
        winner = "Buying" if diff[-1] > 0 else "Renting" if diff[-1] < 0 else "It's a tie"
        st.info(f"⚖️ No crossover in this horizon — **{winner}** leads the whole way.")

    st.markdown("#### Home value vs. mortgage balance")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=yrs, y=det["home_path"], name="Home value",
                              line=dict(color="#f59e0b")))
    fig2.add_trace(go.Scatter(x=yrs, y=det["balance_path"], name="Mortgage balance",
                              line=dict(color="#ef4444")))
    fig2.add_trace(go.Scatter(x=yrs, y=det["home_path"] - det["balance_path"],
                              name="Home equity", fill="tozeroy",
                              line=dict(color="#3b82f6", dash="dot")))
    fig2.update_layout(height=380, xaxis_title="Year", yaxis_title=unit,
                       hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, width='stretch')

    paid = np.where(det["balance_path"] <= 1.0)[0]
    if len(paid):
        extra_note = ("accelerated by your extra principal payments"
                      if extra_principal_pct > 0 else "scheduled payments only")
        st.info(f"🏁 On the expected path the mortgage is fully paid off around "
                f"**year {paid[0] / 12:.1f}** ({extra_note}). After payoff, the "
                "freed-up payment flows straight into investments.")

# ===========================================================================
# TAB 2 — Risk / distribution (Monte Carlo)
# ===========================================================================
with tab_dist:
    st.markdown(f"#### Distribution of outcomes across {n_sims:,} Monte Carlo simulations")
    st.caption("Each simulation draws a different random path for stock returns and "
               "home appreciation (correlated), reflecting real-world uncertainty.")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=owner_mc, name="🏠 Buy", opacity=0.6,
                               marker_color="#2563eb", nbinsx=60))
    fig.add_trace(go.Histogram(x=renter_mc, name="📈 Rent + invest", opacity=0.6,
                               marker_color="#16a34a", nbinsx=60))
    fig.add_vline(x=o_mean, line=dict(color="#2563eb", dash="dash"))
    fig.add_vline(x=r_mean, line=dict(color="#16a34a", dash="dash"))
    fig.update_layout(barmode="overlay", height=430, xaxis_title=f"Terminal net worth ({unit})",
                      yaxis_title="Simulations", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')

    def stats_row(arr):
        return dict(
            Mean=arr.mean(), Median=np.median(arr),
            **{"Std dev (risk)": arr.std(),
               "5th pct (bad case)": np.percentile(arr, 5),
               "95th pct (good case)": np.percentile(arr, 95),
               "Sharpe-like (mean/σ)": arr.mean() / arr.std() if arr.std() else np.nan})

    table = pd.DataFrame({"🏠 Buy": stats_row(owner_mc),
                          "📈 Rent + invest": stats_row(renter_mc)}).T
    st.dataframe(table.style.format(fmt), width='stretch')

    st.markdown("#### Expected-value view (return vs. risk)")
    ev = pd.DataFrame({
        "Strategy": ["🏠 Buy", "📈 Rent + invest"],
        "Expected net worth": [o_mean, r_mean],
        "Risk (std dev)": [owner_mc.std(), renter_mc.std()],
        "Downside (5th pct)": [np.percentile(owner_mc, 5), np.percentile(renter_mc, 5)],
    })
    figev = go.Figure()
    figev.add_trace(go.Scatter(
        x=ev["Risk (std dev)"], y=ev["Expected net worth"], mode="markers+text",
        text=ev["Strategy"], textposition="top center",
        marker=dict(size=22, color=["#2563eb", "#16a34a"])))
    figev.update_layout(height=380, xaxis_title="Risk  (σ of outcomes)",
                        yaxis_title=f"Expected net worth ({unit})")
    st.plotly_chart(figev, width='stretch')
    st.caption("Up and to the left is better: more expected wealth, less risk. "
               "This is the classic risk/return trade-off applied to your decision.")

# ===========================================================================
# TAB 3 — Optimizer
# ===========================================================================
with tab_opt:
    st.markdown("#### Optimize your decision variables")
    st.caption("The model is a set of equations, so we can search for the inputs "
               "that maximize your outcome. We grid-search down-payment % and "
               "holding horizon.")

    oc1, oc2 = st.columns(2)
    objective = oc1.radio("Objective", ["Expected wealth (fast)",
                                        "Risk-adjusted: mean − λ·σ (Monte Carlo)"])
    lam = oc2.slider("Risk aversion λ", 0.0, 2.0, 0.5, 0.1,
                     help="Higher λ penalizes uncertain outcomes more heavily.")

    dp_grid = np.array([0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0])
    hz_grid = np.array([3, 5, 7, 10, 15, 20, 25, 30])

    @st.cache_data(show_spinner="Optimizing…")
    def optimize(base, dp_grid, hz_grid, objective, lam):
        Z = np.zeros((len(hz_grid), len(dp_grid)))
        for i, hz in enumerate(hz_grid):
            for j, dp in enumerate(dp_grid):
                q = dict(base, down_pct=float(dp), horizon_years=int(hz))
                if objective.startswith("Expected"):
                    res = deterministic_path(q)
                    Z[i, j] = float(np.atleast_1d(res["owner_terminal"])[0])
                else:
                    res = monte_carlo(q, n_sims=800, seed=7)
                    a = res["owner_terminal"]
                    Z[i, j] = a.mean() - lam * a.std()
        return Z

    Z = optimize(params, dp_grid, hz_grid, objective, lam)
    bi, bj = np.unravel_index(np.argmax(Z), Z.shape)
    best_hz, best_dp = int(hz_grid[bi]), float(dp_grid[bj])

    heat = go.Figure(go.Heatmap(
        z=Z, x=[f"{d*100:.0f}%" for d in dp_grid], y=[str(h) for h in hz_grid],
        colorscale="Viridis", colorbar=dict(title=f"Objective ({unit})")))
    heat.add_trace(go.Scatter(x=[f"{best_dp*100:.0f}%"], y=[str(best_hz)],
                              mode="markers", marker=dict(symbol="star", size=22,
                              color="gold", line=dict(color="black", width=1)),
                              name="Optimum"))
    heat.update_layout(height=430, xaxis_title="Down payment %",
                       yaxis_title="Holding horizon (yrs)", showlegend=False)
    st.plotly_chart(heat, width='stretch')

    st.success(
        f"⭐ **Optimal for the BUY strategy:** put **{best_dp*100:.0f}%** down and "
        f"hold for **{best_hz} years** → objective value **{mfmt(Z[bi, bj])}**. "
        "Compare this against the rent+invest expected value above."
    )
    st.caption("Note: 'maximize the home outcome' is not the same as 'buying beats "
               "renting' — always check the headline comparison too. Lowering the down "
               "payment frees cash to invest but adds PMI and interest; the optimizer "
               "weighs these against each other.")

    st.markdown("#### 🔑 Pay down the mortgage vs. invest the surplus")
    st.caption("Holding down-payment and horizon at your current settings, this "
               "sweeps how much of each month's surplus goes to extra principal.")

    ep_grid = np.linspace(0.0, 1.0, 11)

    @st.cache_data(show_spinner="Sweeping principal-vs-invest…")
    def optimize_extra(base, ep_grid, objective, lam):
        vals = []
        for ep in ep_grid:
            q = dict(base, extra_principal_pct=float(ep))
            if objective.startswith("Expected"):
                vals.append(float(np.atleast_1d(deterministic_path(q)["owner_terminal"])[0]))
            else:
                a = monte_carlo(q, n_sims=800, seed=7)["owner_terminal"]
                vals.append(a.mean() - lam * a.std())
        return np.array(vals)

    ev = optimize_extra(params, ep_grid, objective, lam)
    best_ep = float(ep_grid[np.argmax(ev)])
    figep = go.Figure()
    figep.add_trace(go.Scatter(x=ep_grid * 100, y=ev, mode="lines+markers",
                               line=dict(color="#2563eb", width=3)))
    figep.add_vline(x=best_ep * 100, line=dict(color="gold", dash="dash"))
    figep.update_layout(height=360, xaxis_title="% of monthly surplus → extra principal",
                        yaxis_title=f"Objective ({unit})")
    st.plotly_chart(figep, width='stretch')
    st.success(f"⭐ **Optimal split:** send **{best_ep*100:.0f}%** of your monthly "
               f"surplus to extra mortgage principal and invest the remaining "
               f"**{100-best_ep*100:.0f}%** in the S&P 500.")
    st.caption(f"With the *Expected wealth* objective this usually lands at 0% — "
               f"stocks' ~{stock_mean*100:.0f}% expected return beats the guaranteed "
               f"~{mortgage_rate*100:.1f}% from paying down the loan. Switch to the "
               "*risk-adjusted* objective and raise λ, and the guaranteed mortgage "
               "payoff starts to win because it carries zero risk.")

# ===========================================================================
# TAB 4 — Sensitivity (tornado)
# ===========================================================================
with tab_sens:
    st.markdown("#### What moves the answer most? (tornado sensitivity)")
    st.caption("Each bar shows how the *Buy advantage* (Buy − Rent expected net worth) "
               "changes when one input is dialed to a low vs. high value, all else fixed.")

    base_adv = deterministic_path(params)["owner_terminal"]
    base_adv = float(np.atleast_1d(base_adv)[0]) - float(
        np.atleast_1d(deterministic_path(params)["renter_terminal"])[0])

    # (label, param_key, low, high)
    knobs = [
        ("Home appreciation %/yr", "home_mean", params["home_mean"] - 0.02, params["home_mean"] + 0.02),
        ("S&P 500 return %/yr", "stock_mean", params["stock_mean"] - 0.02, params["stock_mean"] + 0.02),
        ("Mortgage rate", "mortgage_rate", params["mortgage_rate"] - 0.015, params["mortgage_rate"] + 0.015),
        ("Roommate rent $/mo", "roommate_rent", params["roommate_rent"] * 0.5, params["roommate_rent"] * 1.5),
        ("Your rent $/mo", "your_rent", params["your_rent"] * 0.7, params["your_rent"] * 1.3),
        ("Home price", "home_price", params["home_price"] * 0.85, params["home_price"] * 1.15),
        ("Down payment %", "down_pct", max(0.03, params["down_pct"] - 0.15), min(1.0, params["down_pct"] + 0.15)),
        ("Maintenance %/yr", "maintenance", max(0, params["maintenance"] - 0.01), params["maintenance"] + 0.01),
    ]

    rows = []
    for label, key, lo, hi in knobs:
        adv_lo = _adv = None
        r_lo = deterministic_path(dict(params, **{key: lo}))
        r_hi = deterministic_path(dict(params, **{key: hi}))
        adv_lo = float(np.atleast_1d(r_lo["owner_terminal"])[0]) - float(np.atleast_1d(r_lo["renter_terminal"])[0])
        adv_hi = float(np.atleast_1d(r_hi["owner_terminal"])[0]) - float(np.atleast_1d(r_hi["renter_terminal"])[0])
        rows.append((label, adv_lo - base_adv, adv_hi - base_adv))

    rows.sort(key=lambda r: abs(r[1]) + abs(r[2]))
    labels = [r[0] for r in rows]
    figt = go.Figure()
    figt.add_trace(go.Bar(y=labels, x=[r[1] for r in rows], orientation="h",
                          name="Low value", marker_color="#ef4444"))
    figt.add_trace(go.Bar(y=labels, x=[r[2] for r in rows], orientation="h",
                          name="High value", marker_color="#16a34a"))
    figt.update_layout(barmode="relative", height=430,
                       xaxis_title=f"Change in Buy advantage vs. base ({unit})",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(figt, width='stretch')
    st.caption(f"Base-case Buy advantage (mean returns): **{mfmt(base_adv)}**. "
               "Longest bars = the assumptions your decision is most sensitive to — "
               "pin those down first.")

# ===========================================================================
# TAB 5 — Data & assumptions
# ===========================================================================
with tab_data:
    st.markdown("""
#### Where the default numbers come from (mid-2026, U.S.)

| Input | Default | Source / basis |
|---|---|---|
| S&P 500 return | **10%/yr** | ~10.6% annualized over the last 100 yrs; ~10.3% over 30 yrs (Slickcharts, Macrotrends). |
| S&P 500 volatility σ | **18%** | Long-run annual std dev ~15–20% (large-cap ≈ 19.8%). |
| Home appreciation | **4.5%/yr** | Case-Shiller nominal long-run ≈ 4% (≈1–2% real); recent YoY has cooled. |
| Home volatility σ | **7%** | Home prices are far less volatile than stocks but not risk-free. |
| Mortgage rate | **6.5%** | Freddie Mac 30-yr fixed ≈ 6.43% (week of Jul 2, 2026). |
| Property tax | **1.1%/yr** | U.S. average effective rate ≈ 1.1% (Tax Foundation / Census). |
| Homeowners insurance | **0.6%/yr** | ≈ 0.6–0.7% of value; ~\$2,500/yr on \$400k dwelling (NerdWallet/Forbes). |
| Maintenance | **1.0%/yr** | Classic 1% rule (range 1–3% commonly cited). |
| PMI | **0.7%/yr** | Typical 0.5–1.0% of loan when down payment < 20%. |
| Capital-gains tax (stocks) | **15%** | Long-term LTCG bracket for most filers (0/15/20%). |
| Home-sale exclusion | **\$250k** | IRC §121 exclusion: \$250k single / \$500k married. |
| SALT cap | **\$10k** | State-and-local-tax itemized deduction cap. |
| Inflation | **3%/yr** | Long-run CPI norm. |

**Method.** Each month both scenarios draw from the *same* disposable income
(take-home pay − living expenses) and pay their housing cost; whatever is left is
the investable surplus. RENT invests its whole surplus in the S&P 500; BUY splits
its surplus between extra mortgage principal and the S&P 500 by your chosen ratio.
Because the available cash is identical, the comparison is fair. Stock returns and
home appreciation are drawn from correlated normal distributions (mean, σ, ρ),
giving a full **expected value + risk** distribution rather than a single guess.
Terminal wealth is **after** selling costs, capital-gains taxes, and the
home-sale exclusion.

**Caveats / simplifications.**
- The mortgage-interest + property-tax deduction is modeled simply (deductible ×
  marginal rate, capped by SALT); it ignores the standard-deduction threshold,
  so it can slightly *overstate* the owner's tax benefit. Uncheck it to be
  conservative.
- Rental income uses a single *effective* tax rate to stand in for
  depreciation/expense deductions.
- Returns are modeled as normal & i.i.d. by year (no fat tails, no mean
  reversion, no sequence-of-returns beyond the random draw).
- This is a planning tool, **not financial advice.** Verify current rates for
  your situation before acting.
""")
    st.markdown("**Sources:** Slickcharts, Macrotrends, S&P/Case-Shiller (FRED), "
                "Freddie Mac PMMS, Tax Foundation, NerdWallet, Forbes Advisor, IRS §121.")

st.divider()
st.caption("Educational planning tool — not financial advice. Model returns are "
           "assumptions, not guarantees.")
