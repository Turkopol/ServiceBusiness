
# GAIB ServiceSim — Hotel Business Simulation (Streamlit MVP)

A classroom-friendly business simulation inspired by service-management mechanics (two-season demand, wholesale advance vs. walk-in, HR, facilities, marketing, simple financing).  
Students play in **rounds** making decisions and immediately see **revenues, costs, profit, ROCE, EPS proxy**, etc.

## Features
- Summer/Winter seasonality
- Walk-in pricing & marketing with elasticity and attractiveness composite
- Wholesale **advance sales** for +1 and +2 rounds with price-quantity & credit-term effects
- HR (permanent vs. temporary), training-driven competence, staffing sufficiency
- Facilities condition with wear & maintenance
- Cost-saving levers (ops/admin)
- Simple financing: loan delta, dividends, prime-rate interest
- Results dashboard & CSV export
- Instructor market JSON export/import to keep conditions identical across teams

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the URL printed in the terminal (usually http://localhost:8501).

## Classroom Workflow
1. Instructor sets market parameters in **Instructor / Market Setup** and clicks **Download market_config.json**.  
2. Each team imports the same JSON on their instance to play under identical conditions.  
3. Teams go to **Decisions**, commit rounds (5–8 rounds typical).  
4. Teams export **Results CSV** and submit for grading/leaderboard.

## Notes
- This is a minimal MVP designed for pedagogy and extensibility.
- For multi-team competition on a single server, add a lightweight database (e.g., SQLite or Supabase) and a **market/universe** key.
- The financial model is intentionally simplified; you can deepen it (taxes, depreciation, detailed balance sheet) as needed.
