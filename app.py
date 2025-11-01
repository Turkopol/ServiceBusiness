
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import sqlite3, hashlib, random
from typing import Dict, Any, List, Optional, Tuple

st.set_page_config(page_title="GAIB ServiceSim (Hotel) — Protected", layout="wide")

# =============== Helpers & Config ===============
def clamp(x, lo, hi): return max(lo, min(hi, x))
def currency(x): return f"€{x:,.0f}"
def season_of_round(r): return "Summer" if (r % 2 == 1) else "Winter"

INSTRUCTOR_PASSWORD = st.secrets.get("INSTRUCTOR_PASSWORD", "")
DEFAULT_MARKET_ID = st.secrets.get("MARKET_ID", "MKT-2025-S1")
DEFAULT_TEAM_ID   = st.secrets.get("TEAM_ID", "TEAM-ALPHA")

def seeded_noise(seed_key: str, sd: float) -> float:
    r = int(hashlib.sha256(seed_key.encode()).hexdigest(), 16) % (2**32 - 1)
    rng = random.Random(r)
    u1, u2 = max(rng.random(), 1e-9), rng.random()
    z = ((-2*np.log(u1))**0.5) * np.cos(2*np.pi*u2)  # ~N(0,1)
    return z * sd

# =============== Minimal Kalıcı Depo (SQLite) ===============
DB_PATH = "sim.db"
def db_conn(): return sqlite3.connect(DB_PATH, check_same_thread=False)

def db_init():
    conn = db_conn(); cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS results(
        market_id TEXT, team_id TEXT, round_no INTEGER,
        payload_json TEXT,
        PRIMARY KEY (market_id, team_id, round_no)
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pipeline(
        market_id TEXT, team_id TEXT, round_to_deliver INTEGER,
        nights INTEGER, price REAL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS markets(
        market_id TEXT PRIMARY KEY,
        round_no INTEGER NOT NULL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS teams(
        market_id TEXT, team_id TEXT,
        state_json TEXT,
        PRIMARY KEY (market_id, team_id)
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS decisions(
        market_id TEXT, team_id TEXT, round_no INTEGER,
        decisions_json TEXT,
        PRIMARY KEY (market_id, team_id, round_no)
    )""")
    conn.commit(); conn.close()

def db_get_market_round(market_id: str) -> int:
    conn=db_conn(); cur=conn.cursor()
    cur.execute("SELECT round_no FROM markets WHERE market_id=?", (market_id,))
    row=cur.fetchone()
    if row is None:
        cur.execute("INSERT INTO markets(market_id, round_no) VALUES(?,?)", (market_id, 1))
        conn.commit(); conn.close()
        return 1
    conn.close()
    return int(row[0])

def db_set_market_round(market_id: str, round_no: int):
    conn=db_conn(); cur=conn.cursor()
    cur.execute("INSERT INTO markets(market_id, round_no) VALUES(?,?) ON CONFLICT(market_id) DO UPDATE SET round_no=excluded.round_no",
                (market_id, round_no))
    conn.commit(); conn.close()

def db_upsert_team_state(market_id: str, team_id: str, state: Dict[str,Any]):
    conn=db_conn(); cur=conn.cursor()
    cur.execute("""INSERT INTO teams(market_id, team_id, state_json)
                   VALUES(?,?,?)
                   ON CONFLICT(market_id, team_id) DO UPDATE SET state_json=excluded.state_json""",
                (market_id, team_id, json.dumps(state)))
    conn.commit(); conn.close()

def db_get_team_state(market_id: str, team_id: str) -> Optional[Dict[str,Any]]:
    conn=db_conn(); cur=conn.cursor()
    cur.execute("SELECT state_json FROM teams WHERE market_id=? AND team_id=?", (market_id, team_id))
    row=cur.fetchone(); conn.close()
    return json.loads(row[0]) if row else None

def db_list_team_ids(market_id: str) -> List[str]:
    conn=db_conn(); cur=conn.cursor()
    cur.execute("SELECT team_id FROM teams WHERE market_id=? ORDER BY team_id", (market_id,))
    out=[r[0] for r in cur.fetchall()]
    conn.close(); return out

def db_upsert_result(market_id: str, team_id: str, round_no: int, row: Dict[str, Any]):
    conn = db_conn(); cur = conn.cursor()
    cur.execute("""
    INSERT INTO results(market_id, team_id, round_no, payload_json)
    VALUES(?,?,?,?)
    ON CONFLICT(market_id, team_id, round_no)
    DO UPDATE SET payload_json=excluded.payload_json
    """, (market_id, team_id, round_no, json.dumps(row)))
    conn.commit(); conn.close()

def db_load_results(market_id: str, team_id: str) -> List[Dict[str, Any]]:
    conn = db_conn(); cur = conn.cursor()
    cur.execute("SELECT payload_json FROM results WHERE market_id=? AND team_id=? ORDER BY round_no",
                (market_id, team_id))
    rows = [json.loads(r[0]) for r in cur.fetchall()]
    conn.close()
    return rows

def db_load_pipeline(market_id: str, team_id: str) -> List[Dict[str, Any]]:
    conn = db_conn(); cur = conn.cursor()
    cur.execute("SELECT round_to_deliver, nights, price FROM pipeline WHERE market_id=? AND team_id=?",
                (market_id, team_id))
    out = [{"round_to_deliver": r, "nights": n, "price": p} for (r,n,p) in cur.fetchall()]
    conn.close()
    return out

def db_replace_pipeline(market_id: str, team_id: str, items: List[Dict[str, Any]]):
    conn = db_conn(); cur = conn.cursor()
    cur.execute("DELETE FROM pipeline WHERE market_id=? AND team_id=?", (market_id, team_id))
    cur.executemany("INSERT INTO pipeline(market_id, team_id, round_to_deliver, nights, price) VALUES(?,?,?,?,?)",
                    [(market_id, team_id, it["round_to_deliver"], it["nights"], it["price"]) for it in items])
    conn.commit(); conn.close()

def db_upsert_decisions(market_id: str, team_id: str, round_no: int, decisions: Dict[str,Any]):
    conn=db_conn(); cur=conn.cursor()
    cur.execute("""INSERT INTO decisions(market_id, team_id, round_no, decisions_json)
                   VALUES(?,?,?,?)
                   ON CONFLICT(market_id, team_id, round_no) DO UPDATE SET decisions_json=excluded.decisions_json""",
                (market_id, team_id, round_no, json.dumps(decisions)))
    conn.commit(); conn.close()

def db_get_decisions(market_id: str, team_id: str, round_no: int) -> Optional[Dict[str,Any]]:
    conn=db_conn(); cur=conn.cursor()
    cur.execute("SELECT decisions_json FROM decisions WHERE market_id=? AND team_id=? AND round_no=?",
                (market_id, team_id, round_no))
    row=cur.fetchone(); conn.close()
    return json.loads(row[0]) if row else None

db_init()

# =============== Varsayılan Market & Team State ===============
DEFAULT_MARKET = {
    "base_demand_summer": 18000, "base_demand_winter": 12000, "competitor_pressure": 0.5,
    "wholesale_price_base": 70, "walkin_ref_price": 120, "price_elasticity": -0.7,
    "marketing_effect": 0.25, "quality_weight": 0.30, "staffing_weight": 0.25,
    "marketing_weight": 0.20, "price_weight": 0.25, "prime_rate_6m": 0.025,
    "wholesale_credit_term_effect": 0.0008, "advance_price_slope": 0.35, "random_noise_sd": 0.04
}

if "market" not in st.session_state: st.session_state.market = DEFAULT_MARKET.copy()
if "team" not in st.session_state:
    st.session_state.team = {
        "team_name":"", "shares_outstanding":100000, "long_term_loans":1_000_000,
        "cash":500_000, "equity":1_500_000, "rooms":60, "condition":85.0,
        "perm_emp":20, "temp_emp":6, "cum_training":0.0,
        "avg_salary_perm":1900, "avg_salary_temp":1200, "last_dividend":0.0
    }
if "round" not in st.session_state: st.session_state.round = 1
if "history" not in st.session_state: st.session_state.history = []

if "market_id" not in st.session_state: st.session_state.market_id = DEFAULT_MARKET_ID
if "team_id"   not in st.session_state: st.session_state.team_id   = DEFAULT_TEAM_ID

# round'ı markets tablosundan başlat
st.session_state.round = db_get_market_round(st.session_state.market_id)

# Pipeline’ı DB’den başlat
if "advance_pipeline" not in st.session_state:
    st.session_state.advance_pipeline = db_load_pipeline(st.session_state.market_id, st.session_state.team_id)

# =============== Simulation Core (Preview & Commit) ===============
def staffing_effective(perm, temp, competence):
    return perm * (1.0 + 0.02 * competence) + temp * (0.75 + 0.01 * competence)

def _simulate(team, market, decisions, advance_pipeline, round_no, noise_value: float) -> Dict[str,Any]:
    rooms = team["rooms"]; condition = team["condition"]; perm = team["perm_emp"]; temp = team["temp_emp"]
    cum_training = team["cum_training"]; cap_nights = rooms * 180
    wear = 3.0
    condition_next = clamp(condition - wear + decisions["maintenance"] / 20_000, 20.0, 100.0)
    cum_training_next = cum_training + decisions["training"] / max(perm + temp, 1)
    competence = min(2.5, np.log1p(cum_training_next/1000.0) * 1.5)
    effective_staff = staffing_effective(perm, temp, competence)
    staffing_ratio = clamp(effective_staff / (cap_nights / 180 / 1.5), 0.5, 1.5)
    marketing_score = 1.0 + market["marketing_effect"] * np.log1p(decisions["marketing"] / 10_000)
    perm_ratio = perm / max(perm + temp, 1)
    quality_score = (condition_next/100.0)*0.7 + perm_ratio*0.3
    rel_price = decisions["walkin_price"] / market["walkin_ref_price"]
    base = market["base_demand_summer"] if season_of_round(round_no) == "Summer" else market["base_demand_winter"]
    competitor = 1.0 - 0.3*market["competitor_pressure"]
    attractiveness = (market["quality_weight"]*quality_score +
                      market["staffing_weight"]*clamp(staffing_ratio/1.0,0.5,1.5) +
                      market["marketing_weight"]*marketing_score +
                      market["price_weight"]*clamp(1.0/rel_price,0.5,1.5))
    attractiveness = clamp(attractiveness, 0.5, 1.5)
    walkin_demand = base * (rel_price ** market["price_elasticity"]) * attractiveness * competitor * (1.0 + noise_value)
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
    return {
        "round_no": round_no,"season": season_of_round(round_no),
        "metrics": {"cap_nights":cap_nights,"walkin_nights":walkin_nights,"walkin_rev":walkin_rev,
            "adv_deliver_nights":adv_nights_now,"adv_deliver_rev":adv_rev_now,"revenue":revenue,"payroll":payroll,
            "variable_costs":variable_costs,"admin_costs":admin_costs,"maintenance_cost":maintenance_cost,
            "training_cost":training_cost,"marketing_cost":marketing_cost,"interest_cost":interest_cost,"ebit":ebit,
            "profit":profit,"cash_next":cash_next,"loans_next":loans_next,"equity_next":equity_next,"eps":eps,
            "share_price":share_price,"roce":roce,"gross_margin":gross_margin,"net_margin":net_margin,
            "gearing":gearing,"asset_turnover":asset_turnover,"condition_next":condition_next,"cum_training_next":cum_training_next},
        "new_adv":[new_adv1,new_adv2],
        "advance_pipeline_now": deliver_now
    }

def simulate_round_preview(team, market, decisions, advance_pipeline, round_no):
    # Önizleme: noise=0.0
    base = _simulate(team, market, decisions, advance_pipeline, round_no, noise_value=0.0)
    return {
        "round_no": round_no, "season": base["season"], "team": team, "decisions": decisions,
        "metrics": base["metrics"], "advance_pipeline_now": base["advance_pipeline_now"], "new_adv": base["new_adv"]
    }

def simulate_round_commit(team, market, decisions, advance_pipeline, round_no, seed_key):
    noise = seeded_noise(seed_key, sd=market["random_noise_sd"])
    base = _simulate(team, market, decisions, advance_pipeline, round_no, noise_value=noise)
    return {
        "round_no": round_no, "season": base["season"], "team": team, "decisions": decisions,
        "metrics": base["metrics"], "advance_pipeline_now": base["advance_pipeline_now"], "new_adv": base["new_adv"]
    }

# =============== UI: Instructor / Market Setup ===============
def page_admin():
    st.header("Instructor / Market Setup")
    st.caption("Configure market conditions, set IDs, collect decisions, and resolve the round for ALL teams.")
    tabs = st.tabs(["Configure","Export / Import","IDs","Resolve Round (All Teams)"])
    # --- Configure
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
        st.success("Market updated in session. Use Export/Import to sync JSON if needed.")
    # --- Export / Import
    with tabs[1]:
        colx,coly = st.columns(2)
        with colx:
            st.subheader("Export Market JSON")
            st.download_button("Download market_config.json", data=json.dumps(st.session_state.market, indent=2),
                               file_name="market_config.json", mime="application/json")
        with coly:
            st.subheader("Import Market JSON")
            up = st.file_uploader("Upload market_config.json", type=["json"])
            if up is not None:
                try:
                    st.session_state.market = json.load(up)
                    st.success("Market config imported into session.")
                except Exception as e:
                    st.error(f"Import failed: {e}")
    # --- IDs
    with tabs[2]:
        st.text_input("Market ID", key="market_id")
        st.text_input("Team ID", key="team_id")
        st.info("These IDs are stored in session and used for results/pipeline keys.")
        st.caption("Teams table is keyed by (market_id, team_id). Use Home page to save/sync each team's initial state.")
    # --- Resolve Round (All Teams)
    with tabs[3]:
        market_id = st.session_state.market_id
        current_round = db_get_market_round(market_id)
        st.subheader(f"Resolve Market Round — {market_id} | Round {current_round} ({season_of_round(current_round)})")
        team_ids = db_list_team_ids(market_id)
        st.write(f"Registered teams in this market: **{', '.join(team_ids) if team_ids else '(none)'}**")
        if st.button("🧮 Resolve this Round for ALL teams", type="primary"):
            if not team_ids:
                st.warning("No teams registered in this market.")
            else:
                resolved, skipped = 0, []
                for tid in team_ids:
                    team_state = db_get_team_state(market_id, tid)
                    if not team_state:
                        skipped.append((tid, "no team state")); continue
                    pipeline = db_load_pipeline(market_id, tid)
                    dec = db_get_decisions(market_id, tid, current_round)
                    if not dec:
                        skipped.append((tid, "no decisions submitted")); continue
                    final = simulate_round_commit(
                        team=team_state.copy(),
                        market=st.session_state.market.copy(),
                        decisions=dec,
                        advance_pipeline=pipeline.copy(),
                        round_no=current_round,
                        seed_key=f"{market_id}-{tid}-{current_round}"
                    )
                    m = final["metrics"]
                    # update team state
                    team_state["cash"]=m["cash_next"]; team_state["equity"]=m["equity_next"]; team_state["long_term_loans"]=m["loans_next"]
                    team_state["condition"]=m["condition_next"]; team_state["cum_training"]=m["cum_training_next"]; team_state["last_dividend"]=dec["dividend"]
                    db_upsert_team_state(market_id, tid, team_state)
                    # update pipeline
                    new_pipe = [x for x in pipeline if x["round_to_deliver"] != final["round_no"]]
                    new_pipe.extend(final["new_adv"])
                    db_replace_pipeline(market_id, tid, new_pipe)
                    # write result
                    log_row = {"market_id": market_id, "team_id": tid,
                               "round": final["round_no"], "season": final["season"],
                               **{f"dec_{k}": v for k,v in dec.items()}, **m}
                    db_upsert_result(market_id, tid, final["round_no"], log_row)
                    resolved += 1
                # advance market round
                db_set_market_round(market_id, current_round + 1)
                st.success(f"Resolved for {resolved} team(s).")
                if skipped:
                    st.warning("Skipped teams (reason): " + ", ".join([f"{t}:{r}" for t,r in skipped]))
                st.experimental_rerun()

def protected_page_admin():
    st.header("Instructor / Market Setup")
    st.caption("🔒 This section is password-protected.")
    if "admin_unlocked" not in st.session_state: st.session_state.admin_unlocked = False
    if not st.session_state.admin_unlocked:
        pw = st.text_input("🔐 Instructor Password", type="password", help="Ask your instructor for access.")
        if st.button("Unlock"):
            if pw == INSTRUCTOR_PASSWORD and INSTRUCTOR_PASSWORD:
                st.session_state.admin_unlocked = True
                st.success("Instructor panel unlocked ✅")
            else:
                st.error("Incorrect password.")
        st.stop()
    page_admin()

# =============== UI: Home / Decisions / Results ===============
def page_home():
    st.title("GAIB ServiceSim — Hotel Scenario (Protected)")
    st.caption("Two-season service simulation with wholesale & retail decisions, HR, facilities, marketing, and financing.")
    st.info(f"Market ID: {st.session_state.market_id} | Team ID: {st.session_state.team_id}")
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
    c5,c6 = st.columns(2)
    if c5.button("💾 Save/Sync Team State to DB"):
        db_upsert_team_state(st.session_state.market_id, st.session_state.team_id, t.copy())
        st.success("Team state saved to DB.")
    if c6.button("⬇️ Load Team State from DB"):
        s = db_get_team_state(st.session_state.market_id, st.session_state.team_id)
        if s: st.session_state.team = s; st.success("Loaded from DB.")
        else: st.info("No saved state for this team.")
    st.info("When ready, go to **Decisions** to submit Round decisions to your instructor.")

def page_decisions():
    market_round = db_get_market_round(st.session_state.market_id)  # authoritative round
    st.header(f"Decisions — Round {market_round} ({season_of_round(market_round)})")
    t = st.session_state.team; m = st.session_state.market
    cap_nights = t["rooms"] * 180
    colA,colB = st.columns(2)
    with colA:
        st.subheader("Sales & Marketing")
        walkin_price = st.number_input("Walk-in Price (€/night)", 40, 500, int(m["walkin_ref_price"]), step=5, key="walkin_price")
        marketing = st.number_input("Marketing Budget (€)", 0, 500_000, 30_000, step=5_000, key="marketing")
        adv_p1 = st.number_input("Advance Sales request for +1 (nights)", 0, cap_nights, 3000, step=100, key="adv_p1")
        adv_p2 = st.number_input("Advance Sales request for +2 (nights)", 0, cap_nights, 2000, step=100, key="adv_p2")
        credit_term_days = st.number_input("Credit Term for Wholesale (days)", 0, 120, 30, step=5, key="credit_term_days")
    with colB:
        st.subheader("Operations & HR")
        maintenance = st.number_input("Maintenance / Renovation Budget (€)", 0, 500_000, 20_000, step=5_000, key="maintenance")
        training = st.number_input("Training Budget (€)", 0, 500_000, 10_000, step=5_000, key="training")
        costsave_ops = st.slider("Cost-saving Effort (Operations)", 0.0, 1.0, 0.2, 0.05, key="costsave_ops")
        costsave_admin = st.slider("Cost-saving Effort (Admin)", 0.0, 1.0, 0.1, 0.05, key="costsave_admin")
        st.subheader("Financing")
        loan_delta = st.number_input("Change in Long-term Loans (Δ€)", -500_000, 500_000, 0, step=10_000, key="loan_delta")
        dividend = st.number_input("Dividends (€ total)", 0, 500_000, 0, step=10_000, key="dividend")

    dec = {
        "walkin_price": walkin_price,"marketing": marketing,"adv_p1": adv_p1,"adv_p2": adv_p2,
        "credit_term_days": credit_term_days,"maintenance": maintenance,"training": training,
        "costsave_ops": float(costsave_ops),"costsave_admin": float(costsave_admin),
        "loan_delta": loan_delta,"dividend": dividend
    }

    preview = simulate_round_preview(
        team=st.session_state.team.copy(),
        market=st.session_state.market.copy(),
        decisions=dec,
        advance_pipeline=st.session_state.advance_pipeline.copy(),
        round_no=market_round
    )
    st.markdown("### Preview (Before Submit)"); show_preview(preview)

    c1,c2 = st.columns(2)
    if c1.button("📤 Submit Decisions to Instructor"):
        # Save decisions for authoritative resolve
        db_upsert_decisions(st.session_state.market_id, st.session_state.team_id, market_round, dec)
        # Ensure team state is persisted
        db_upsert_team_state(st.session_state.market_id, st.session_state.team_id, st.session_state.team.copy())
        st.success("Decisions submitted to instructor for this round.")
    if c2.button("✅ Commit Locally (solo mode)"):
        # Solo/test modu: tek takım kendi turunu hemen çözer (ders demosu için)
        apply_round(preview)  # uses local round and increments markets/round via DB later? We'll keep local
        st.success("Round committed locally. (For class fairness, prefer instructor resolve.)")
        st.balloons()

def show_preview(preview):
    m = preview["metrics"]
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Revenue", currency(m["revenue"])); k2.metric("EBIT", currency(m["ebit"]))
    k3.metric("Profit", currency(m["profit"])); k4.metric("ROCE", f"{m['roce']*100:.1f}%")
    k5.metric("EPS (€/share)", f"{m['eps']:.3f}")
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

# =============== Local Commit (Solo/Test) ===============
def apply_round(preview):
    # Deterministik commit (solo): markets tablosundaki round'u kullan
    market_id = st.session_state.market_id
    team_id = st.session_state.team_id
    current_round = db_get_market_round(market_id)
    dec = preview["decisions"]
    final = simulate_round_commit(
        team=st.session_state.team.copy(),
        market=st.session_state.market.copy(),
        decisions=dec,
        advance_pipeline=st.session_state.advance_pipeline.copy(),
        round_no=current_round,
        seed_key=f"{market_id}-{team_id}-{current_round}"
    )
    m = final["metrics"]
    # State güncelle
    t = st.session_state.team
    t["cash"]=m["cash_next"]; t["equity"]=m["equity_next"]; t["long_term_loans"]=m["loans_next"]
    t["condition"]=m["condition_next"]; t["cum_training"]=m["cum_training_next"]; t["last_dividend"]=dec["dividend"]
    # Pipeline güncelle
    pipeline = db_load_pipeline(market_id, team_id)
    new_pipe = [x for x in pipeline if x["round_to_deliver"] != final["round_no"]]
    new_pipe.extend(final["new_adv"])
    st.session_state.advance_pipeline = new_pipe
    db_replace_pipeline(market_id, team_id, new_pipe)
    # Sonuç yaz
    log_row = {"market_id": market_id, "team_id": team_id,
               "round": final["round_no"], "season": final["season"],
               **{f"dec_{k}": v for k,v in dec.items()}, **m}
    db_upsert_result(market_id, team_id, final["round_no"], log_row)
    # Team state persist
    db_upsert_team_state(market_id, team_id, t.copy())
    # Round advance (solo/test)
    db_set_market_round(market_id, current_round + 1)

def page_results():
    st.header("Results & Reports")
    rows_db = db_load_results(st.session_state.market_id, st.session_state.team_id)
    hist = pd.DataFrame(rows_db if rows_db else st.session_state.history)
    if hist.empty:
        st.info("No rounds committed yet."); return
    st.dataframe(hist, use_container_width=True)
    st.download_button("Download Results CSV",
                       data=hist.to_csv(index=False).encode("utf-8"),
                       file_name=f"{st.session_state.team['team_name'].replace(' ','_')}_results.csv",
                       mime="text/csv")
    st.markdown("### Performance Over Time")
    ch1,ch2 = st.columns(2)
    with ch1:
        cols = [c for c in ["revenue","ebit","profit"] if c in hist.columns]
        if cols: st.line_chart(hist.set_index("round")[cols])
    with ch2:
        cols = [c for c in ["roce","net_margin","asset_turnover"] if c in hist.columns]
        if cols: st.line_chart(hist.set_index("round")[cols])

# =============== Router ===============
pages = {
    "Home (Team Setup)": page_home,
    "Decisions": page_decisions,
    "Results": page_results,
    "Instructor / Market Setup": protected_page_admin
}
choice = st.sidebar.radio("Navigate", list(pages.keys()))
pages[choice]()
st.sidebar.markdown("---")
st.sidebar.caption("GAIB ServiceSim — Streamlit MVP (Deterministic + Multi-team Resolve)")

