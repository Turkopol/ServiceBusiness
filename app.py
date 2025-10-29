
# GAIB ServiceSim — Streamlit Business Simulation (Hotel Scenario) [Protected Instructor Page]
import streamlit as st
import pandas as pd
import numpy as np
import json

st.set_page_config(page_title="GAIB ServiceSim (Hotel) — Protected", layout="wide")

def clamp(x, lo, hi): return max(lo, min(hi, x))
def currency(x): return f"€{x:,.0f}"
def season_of_round(r): return "Summer" if (r % 2 == 1) else "Winter"

DEFAULT_MARKET = {
    "base_demand_summer": 18000, "base_demand_winter": 12000, "competitor_pressure": 0.5,
    "wholesale_price_base": 70, "walkin_ref_price": 120, "price_elasticity": -0.7,
    "marketing_effect": 0.25, "quality_weight": 0.30, "staffing_weight": 0.25,
    "marketing_weight": 0.20, "price_weight": 0.25, "prime_rate_6m": 0.025,
    "wholesale_credit_term_effect": 0.0008, "advance_price_slope": 0.35, "random_noise_sd": 0.04
}
if "market" not in st.session_state: st.session_state.market = DEFAULT_MARKET.copy()
if "team" not in st.session_state:
    st.session_state.team = {"team_name":"", "shares_outstanding":100000, "long_term_loans":1_000_000,
    "cash":500_000, "equity":1_500_000, "rooms":60, "condition":85.0, "perm_emp":20, "temp_emp":6,
    "cum_training":0.0, "avg_salary_perm":1900, "avg_salary_temp":1200, "last_dividend":0.0}
if "round" not in st.session_state: st.session_state.round = 1
if "history" not in st.session_state: st.session_state.history = []
if "advance_pipeline" not in st.session_state: st.session_state.advance_pipeline = []

def page_admin():
    st.header("Instructor / Market Setup")
    st.caption("Configure market conditions and prime rate. Export/Import JSON to synchronize across teams.")
    tabs = st.tabs(["Configure","Export / Import"])
    with tabs[0]:
        m = st.session_state.market
        col1,col2,col3 = st.columns(3)
        with col1:
            m["base_demand_summer"] = st.number_input("Base Demand (Summer, room-nights)", 5000, 100000, m["base_demand_summer"], step=500)
            m["base_demand_winter"] = st.number_input("Base Demand (Winter, room-nights)", 5000, 100000, m["base_demand_winter"], step=500)
            m["competitor_pressure"] = st.slider("Competitor Pressure", 0.0, 1.0, float(m["competitor_pressure"]), 0.05)
        with col2:
            m["wholesale_price_base"] = st.number_input("Wholesale Price Base (€)", 30, 300, m["wholesale_price_base"], step=5)
            m["walkin_ref_price"] = st.number_input("Walk-in Reference Price (€)", 50, 500, m["walkin_ref_price"], step=5)
            m["price_elasticity"] = st.slider("Walk-in Price Elasticity", -2.0, -0.1, float(m["price_elasticity"]), 0.05)
        with col3:
            m["prime_rate_6m"] = st.number_input("Prime Rate (6-month)", 0.0, 0.2, m["prime_rate_6m"], step=0.005, format="%.3f")
            m["random_noise_sd"] = st.number_input("Random Shock SD", 0.0, 0.2, m["random_noise_sd"], step=0.005, format="%.3f")
        st.markdown("**Attractiveness Weights**")
        colw1,colw2,colw3,colw4 = st.columns(4)
        with colw1: m["quality_weight"] = st.slider("Quality Weight", 0.0, 1.0, float(m["quality_weight"]), 0.05)
        with colw2: m["staffing_weight"] = st.slider("Staffing Weight", 0.0, 1.0, float(m["staffing_weight"]), 0.05)
        with colw3: m["marketing_weight"] = st.slider("Marketing Weight", 0.0, 1.0, float(m["marketing_weight"]), 0.05)
        with colw4: m["price_weight"] = st.slider("Price Weight", 0.0, 1.0, float(m["price_weight"]), 0.05)
        st.markdown("**Wholesale Dynamics**")
        colh1,colh2 = st.columns(2)
        with colh1: m["wholesale_credit_term_effect"] = st.number_input("Credit Term Effect (per day)", 0.0, 0.01, m["wholesale_credit_term_effect"], step=0.0001, format="%.4f")
        with colh2: m["advance_price_slope"] = st.number_input("Advance Price Slope", 0.0, 1.5, m["advance_price_slope"], step=0.05)
        st.success("Market updated in session. Remember to Export JSON to share with teams.")
    with tabs[1]:
        colx,coly = st.columns(2)
        with colx:
            st.subheader("Export Market JSON")
            st.download_button("Download market_config.json", data=json.dumps(st.session_state.market, indent=2), file_name="market_config.json", mime="application/json")
        with coly:
            st.subheader("Import Market JSON")
            up = st.file_uploader("Upload market_config.json", type=["json"])
            if up is not None:
                try:
                    st.session_state.market = json.load(up)
                    st.success("Market config imported into session.")
                except Exception as e:
                    st.error(f"Import failed: {e}")

INSTRUCTOR_PASSWORD = "gaib2025"

def protected_page_admin():
    st.header("Instructor / Market Setup")
    st.caption("🔒 This section is password-protected.")
    if "admin_unlocked" not in st.session_state: st.session_state.admin_unlocked = False
    if not st.session_state.admin_unlocked:
        pw = st.text_input("🔐 Instructor Password", type="password", help="Ask your instructor for access.")
        if st.button("Unlock"):
            if pw == INSTRUCTOR_PASSWORD:
                st.session_state.admin_unlocked = True
                st.success("Instructor panel unlocked ✅")
            else:
                st.error("Incorrect password.")
        st.stop()
    page_admin()

def page_home():
    st.title("GAIB ServiceSim — Hotel Scenario (Protected)")
    st.caption("Two-season service simulation with wholesale & retail decisions, HR, facilities, marketing, and financing.")
    st.markdown("Use the sidebar to navigate between **Decisions**, **Results**, and **Instructor / Market Setup**.")
    t = st.session_state.team
    st.subheader("Team Initialization")
    t["team_name"] = st.text_input("Team Name", value=t["team_name"] or "Team Alpha")
    cols = st.columns(3)
    with cols[0]:
        t["rooms"] = st.number_input("Rooms (capacity)", 10, 500, t["rooms"], step=5)
        t["condition"] = st.slider("Facility Condition (0–100)", 20.0, 100.0, float(t["condition"]), 1.0)
    with cols[1]:
        t["perm_emp"] = st.number_input("Permanent Employees", 1, 500, t["perm_emp"], step=1)
        t["temp_emp"] = st.number_input("Temporary Employees", 0, 500, t["temp_emp"], step=1)
    with cols[2]:
        t["avg_salary_perm"] = st.number_input("Avg Salary (Permanent, per 6 months €)", 500, 5000, t["avg_salary_perm"], step=50)
        t["avg_salary_temp"] = st.number_input("Avg Salary (Temporary, per 6 months €)", 300, 4000, t["avg_salary_temp"], step=50)
    st.markdown("---")
    st.subheader("Financial Position")
    c1,c2,c3,c4 = st.columns(4)
    with c1: t["cash"] = st.number_input("Cash (€)", 0, 10_000_000, t["cash"], step=10_000)
    with c2: t["equity"] = st.number_input("Equity (€)", 0, 10_000_000, t["equity"], step=10_000)
    with c3: t["long_term_loans"] = st.number_input("Long-term Loans (€)", 0, 10_000_000, t["long_term_loans"], step=10_000)
    with c4: t["shares_outstanding"] = st.number_input("Shares Outstanding", 1, 10_000_000, t["shares_outstanding"], step=1000)
    st.info("When ready, go to **Decisions** to play Round 1.")

def page_decisions():
    st.header(f"Decisions — Round {st.session_state.round} ({season_of_round(st.session_state.round)})")
    t = st.session_state.team; m = st.session_state.market
    cap_nights = t["rooms"] * 180
    colA,colB = st.columns(2)
    with colA:
        st.subheader("Sales & Marketing")
        walkin_price = st.number_input("Walk-in Price (€/night)", 40, 500, int(m["walkin_ref_price"]), step=5)
        marketing = st.number_input("Marketing Budget (€)", 0, 500_000, 30_000, step=5_000)
        adv_p1 = st.number_input("Advance Sales request for +1 (nights)", 0, cap_nights, 3000, step=100)
        adv_p2 = st.number_input("Advance Sales request for +2 (nights)", 0, cap_nights, 2000, step=100)
        credit_term_days = st.number_input("Credit Term for Wholesale (days)", 0, 120, 30, step=5)
    with colB:
        st.subheader("Operations & HR")
        maintenance = st.number_input("Maintenance / Renovation Budget (€)", 0, 500_000, 20_000, step=5_000)
        training = st.number_input("Training Budget (€)", 0, 500_000, 10_000, step=5_000)
        costsave_ops = st.slider("Cost-saving Effort (Operations)", 0.0, 1.0, 0.2, 0.05)
        costsave_admin = st.slider("Cost-saving Effort (Admin)", 0.0, 1.0, 0.1, 0.05)
        st.subheader("Financing")
        loan_delta = st.number_input("Change in Long-term Loans (Δ€)", -500_000, 500_000, 0, step=10_000)
        dividend = st.number_input("Dividends (€ total)", 0, 500_000, 0, step=10_000)

    preview = simulate_round_preview(
        team=st.session_state.team.copy(),
        market=st.session_state.market.copy(),
        decisions={
            "walkin_price": walkin_price,"marketing": marketing,"adv_p1": adv_p1,"adv_p2": adv_p2,
            "credit_term_days": credit_term_days,"maintenance": maintenance,"training": training,
            "costsave_ops": float(costsave_ops),"costsave_admin": float(costsave_admin),
            "loan_delta": loan_delta,"dividend": dividend
        },
        advance_pipeline=st.session_state.advance_pipeline.copy(),
        round_no=st.session_state.round
    )
    st.markdown("### Preview (Before Commit)"); show_preview(preview)
    if st.button("✅ Commit Decisions for this Round", type="primary"):
        apply_round(preview); st.success("Round committed. Check the Results page or proceed to next round."); st.balloons()

def staffing_effective(perm, temp, competence):
    return perm * (1.0 + 0.02 * competence) + temp * (0.75 + 0.01 * competence)

def simulate_round_preview(team, market, decisions, advance_pipeline, round_no):
    rooms = team["rooms"]; condition = team["condition"]; perm = team["perm_emp"]; temp = team["temp_emp"]
    cum_training = team["cum_training"]
    cap_nights = rooms * 180
    wear = 3.0; condition_next = clamp(condition - wear + decisions["maintenance"] / 20_000, 20.0, 100.0)
    cum_training_next = cum_training + decisions["training"] / max(perm + temp, 1)
    competence = min(2.5, np.log1p(cum_training_next/1000.0) * 1.5)
    effective_staff = staffing_effective(perm, temp, competence)
    staffing_ratio = clamp(effective_staff / (cap_nights / 180 / 1.5), 0.5, 1.5)
    marketing_score = 1.0 + market["marketing_effect"] * np.log1p(decisions["marketing"] / 10_000)
    perm_ratio = perm / max(perm + temp, 1)
    quality_score = (condition_next/100.0)*0.7 + perm_ratio*0.3
    rel_price = decisions["walkin_price"] / market["walkin_ref_price"]
    base = market["base_demand_summer"] if season_of_round(round_no) == "Summer" else market["base_demand_winter"]
    noise = np.random.normal(0, market["random_noise_sd"]); competitor = 1.0 - 0.3*market["competitor_pressure"]
    attractiveness = (market["quality_weight"]*quality_score + market["staffing_weight"]*clamp(staffing_ratio/1.0,0.5,1.5) +
                      market["marketing_weight"]*marketing_score + market["price_weight"]*clamp(1.0/rel_price,0.5,1.5))
    attractiveness = clamp(attractiveness, 0.5, 1.5)
    walkin_demand = base * (rel_price ** market["price_elasticity"]) * attractiveness * competitor * (1.0 + noise)
    walkin_nights = int(min(cap_nights, max(0, walkin_demand)))
    def wholesale_price(qty_nights):
        q_ratio = qty_nights / max(cap_nights, 1)
        discount = market["advance_price_slope"] * q_ratio + market["wholesale_credit_term_effect"] * decisions["credit_term_days"]
        return max(30, market["wholesale_price_base"] * (1.0 - discount))
    price_adv1 = wholesale_price(decisions["adv_p1"]); price_adv2 = wholesale_price(decisions["adv_p2"])
    new_adv1 = {"round_to_deliver": round_no + 1, "nights": int(min(decisions["adv_p1"], cap_nights)), "price": price_adv1}
    new_adv2 = {"round_to_deliver": round_no + 2, "nights": int(min(decisions["adv_p2"], cap_nights)), "price": price_adv2}
    deliver_now = [x for x in advance_pipeline if x["round_to_deliver"] == round_no]
    adv_nights_now = sum(x["nights"] for x in deliver_now); adv_rev_now = sum(x["nights"]*x["price"] for x in deliver_now)
    total_nights_pre = adv_nights_now + walkin_nights
    if total_nights_pre > cap_nights:
        overflow = total_nights_pre - cap_nights; walkin_nights = max(0, walkin_nights - overflow)
    walkin_rev = walkin_nights * decisions["walkin_price"]; revenue = walkin_rev + adv_rev_now
    payroll = team["avg_salary_perm"] * perm + team["avg_salary_temp"] * temp
    var_cost_per_night = 18.0 * (1.0 - 0.25*decisions["costsave_ops"])
    occ_nights = walkin_nights + min(adv_nights_now, cap_nights - walkin_nights)
    variable_costs = var_cost_per_night * occ_nights; admin_costs = 80_000 * (1.0 - 0.25*decisions["costsave_admin"])
    maintenance_cost = decisions["maintenance"]; training_cost = decisions["training"]; marketing_cost = decisions["marketing"]
    interest_cost = (team["long_term_loans"] + max(decisions["loan_delta"], 0)) * market["prime_rate_6m"]
    dividend = decisions["dividend"]
    opex = payroll + variable_costs + admin_costs + maintenance_cost + training_cost + marketing_cost
    ebit = revenue - opex; profit = ebit - interest_cost - dividend
    cash_next = team.get("cash",500_000) + profit + decisions["loan_delta"] - dividend
    loans_next = team["long_term_loans"] + decisions["loan_delta"]
    equity_next = max(0, team.get("equity",1_500_000) + profit - dividend)
    eps = profit / max(1, team["shares_outstanding"]); pe = 12.0; share_price = max(0.01, pe * max(0.01, eps))
    total_assets = loans_next + equity_next; current_liab = 0
    roce = (ebit) / max(1.0, (total_assets - current_liab))
    gross_margin = (revenue - variable_costs) / max(1.0, revenue) if revenue>0 else 0.0
    net_margin = profit / max(1.0, revenue) if revenue>0 else 0.0
    gearing = (loans_next - cash_next) / max(1.0, equity_next) if equity_next>0 else 1.0
    asset_turnover = revenue / max(1.0, total_assets)
    preview = {"round_no": round_no,"season": season_of_round(round_no),"team": team,"decisions": decisions,
        "metrics": {"cap_nights": cap_nights,"walkin_nights": walkin_nights,"walkin_rev": walkin_rev,
            "adv_deliver_nights": adv_nights_now,"adv_deliver_rev": adv_rev_now,"revenue": revenue,"payroll": payroll,
            "variable_costs": variable_costs,"admin_costs": admin_costs,"maintenance_cost": maintenance_cost,
            "training_cost": training_cost,"marketing_cost": marketing_cost,"interest_cost": interest_cost,"ebit": ebit,
            "profit": profit,"cash_next": cash_next,"loans_next": loans_next,"equity_next": equity_next,"eps": eps,
            "share_price": share_price,"roce": roce,"gross_margin": gross_margin,"net_margin": net_margin,
            "gearing": gearing,"asset_turnover": asset_turnover,"condition_next": condition_next,"cum_training_next": cum_training_next},
        "advance_pipeline_now": deliver_now,"new_adv": [new_adv1, new_adv2]}
    return preview

def show_preview(preview):
    m = preview["metrics"]
    kpi1,kpi2,kpi3,kpi4,kpi5 = st.columns(5)
    kpi1.metric("Revenue", currency(m["revenue"])); kpi2.metric("EBIT", currency(m["ebit"]))
    kpi3.metric("Profit", currency(m["profit"])); kpi4.metric("ROCE", f"{m['roce']*100:.1f}%")
    kpi5.metric("EPS (€/share)", f"{m['eps']:.3f}")
    st.markdown("#### Capacity & Demand")
    c1,c2,c3 = st.columns(3)
    c1.write(f"Capacity (nights): **{m['cap_nights']:,}**")
    c2.write(f"Walk-in nights: **{m['walkin_nights']:,}** → {currency(m['walkin_rev'])}")
    c3.write(f"Advance delivered nights: **{m['adv_deliver_nights']:,}** → {currency(m['adv_deliver_rev'])}")
    st.markdown("#### Cost Breakdown")
    cost_df = pd.DataFrame({"Cost Item":["Payroll","Variable Costs","Admin","Maintenance","Training","Marketing","Interest"],
                            "Amount (€)":[m["payroll"],m["variable_costs"],m["admin_costs"],m["maintenance_cost"],m["training_cost"],m["marketing_cost"],m["interest_cost"]]})
    st.dataframe(cost_df, use_container_width=True)
    st.markdown("#### Financial Position (Post-Round)")
    c4,c5,c6 = st.columns(3)
    c4.write(f"Cash: **{currency(m['cash_next'])}**"); c5.write(f"Loans: **{currency(m['loans_next'])}**"); c6.write(f"Equity: **{currency(m['equity_next'])}**")
    c4.write(f"Net Margin: **{m['net_margin']*100:.1f}%**"); c5.write(f"Gearing: **{m['gearing']*100:.1f}%**"); c6.write(f"Asset Turnover: **{m['asset_turnover']:.2f}x**")
    st.markdown("#### Ops & Quality")
    c7,c8 = st.columns(2)
    c7.write(f"Condition next: **{m['condition_next']:.1f}/100**"); c8.write(f"Cumulative Training next: **{m['cum_training_next']:.0f}**")

def apply_round(preview):
    m = preview["metrics"]; dec = preview["decisions"]
    t = st.session_state.team
    t["cash"] = m["cash_next"]; t["equity"] = m["equity_next"]; t["long_term_loans"] = m["loans_next"]
    t["condition"] = m["condition_next"]; t["cum_training"] = m["cum_training_next"]; t["last_dividend"] = dec["dividend"]
    new_pipe = [x for x in st.session_state.advance_pipeline if x["round_to_deliver"] != preview["round_no"]]
    new_pipe.extend(preview["new_adv"]); st.session_state.advance_pipeline = new_pipe
    log_row = {"round": preview["round_no"], "season": preview["season"], **{f"dec_{k}": v for k,v in dec.items()}, **m}
    st.session_state.history.append(log_row); st.session_state.round += 1

def page_results():
    st.header("Results & Reports")
    import pandas as pd
    hist = pd.DataFrame(st.session_state.history)
    if hist.empty: st.info("No rounds committed yet."); return
    st.dataframe(hist, use_container_width=True)
    st.download_button("Download Results CSV", data=hist.to_csv(index=False).encode("utf-8"),
                       file_name=f"{st.session_state.team['team_name'].replace(' ','_')}_results.csv", mime="text/csv")
    st.markdown("### Performance Over Time")
    ch1,ch2 = st.columns(2)
    with ch1: st.line_chart(hist.set_index("round")[["revenue","ebit","profit"]])
    with ch2: st.line_chart(hist.set_index("round")[["roce","net_margin","asset_turnover"]])

pages = {"Home (Team Setup)": page_home, "Decisions": page_decisions, "Results": page_results, "Instructor / Market Setup": protected_page_admin}
choice = st.sidebar.radio("Navigate", list(pages.keys()))
pages[choice]()
st.sidebar.markdown("---"); st.sidebar.caption("GAIB ServiceSim — Streamlit MVP (Protected)")
