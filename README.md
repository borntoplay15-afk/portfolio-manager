# 📈 Portfolio Manager

A portfolio app **anyone** can use: upload your holdings, answer a 5-question risk
quiz, and get a **personalised** Markowitz efficient frontier plus explicit
**invest / divest** guidance. Also includes broker recommendations, a value
screener, and a single conviction buy list. Built around a **core & satellite** strategy.

## ✨ What it does
1. **Load a portfolio** — one-click demo, **upload a CSV** (`ticker, quantity, avg_cost`),
   or type holdings in manually. GBP / USD / EUR base currency.
2. **Risk questionnaire** — 5 quick questions score you Conservative / Moderate /
   Aggressive (with a manual override slider).
3. **Personalised efficient frontier** — the recommended point on the frontier
   **moves with your risk profile** (Conservative → min-variance · Moderate →
   max-Sharpe · Aggressive → max-utility), not just the concentration cap.
4. **Invest / divest** — given your current portfolio and risk target, it shows
   exactly where to add and where to trim, in money terms, plus how to deploy new cash.

## ▶️ How to launch (every time)

1. Go to **https://colab.research.google.com**
2. **File → Open notebook → Upload** → choose `portfolio_app_launcher.ipynb`
3. **Runtime → Run all** (wait ~90 seconds)
4. The last cell prints a link like `https://xxxx.trycloudflare.com` — **click it**
5. Keep the Colab tab open. The link stops working when Colab closes (~90 min idle),
   and you get a **new link each time** you run it.

## 🤖 Or just ask Claude

In Claude Code, type **`/portfolio-manager`** (or just "check my portfolio" / "recommend a stock").
The skill knows your holdings, strategy, and how to launch or change the app.

## 🛠️ Files

| File | What it is |
|------|-----------|
| `app.py` | The whole app (UI + quant engine) |
| `portfolio_app_launcher.ipynb` | The Colab notebook you upload — it contains the app |
| `build_notebook.mjs` | Rebuilds the notebook after editing `app.py` (`node build_notebook.mjs`) |
| `requirements.txt` | Python dependencies |

## ✏️ To change the app
Edit `app.py` → run `node build_notebook.mjs` → re-upload the notebook to Colab → Run all.
(Or just ask Claude to make the change.)

## 💡 Strategy framework (core & satellite)
- **CORE** — broad low-cost ETFs (e.g. VWRP) + defensive ballast (gilts e.g. IGLT, gold e.g. SGLN)
- **SATELLITE** — individual value/quality picks (~5% each) + a small ring-fenced multibagger sleeve
- General principle: keep the core diversified, size satellites small, and trim any oversized single-name concentration.
