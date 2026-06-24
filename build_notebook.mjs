// Builds a self-contained Colab launcher notebook that embeds app.py via %%writefile.
import { readFileSync, writeFileSync, existsSync } from "node:fs";

const app = readFileSync(new URL("./app.py", import.meta.url), "utf8");

// Private real-portfolio override — embedded into the LOCAL notebook only.
// app.py imports HOLDINGS from this if present. It is gitignored, so the public
// repo/deploy never sees it; the notebook stays on your machine.
const privUrl = new URL("./_private_holdings.py", import.meta.url);
const priv = existsSync(privUrl) ? readFileSync(privUrl, "utf8") : null;

const codeCell = (src) => ({
  cell_type: "code",
  execution_count: null,
  metadata: {},
  outputs: [],
  source: src.split("\n").map((l, i, a) => (i === a.length - 1 ? l : l + "\n")),
});
const mdCell = (src) => ({
  cell_type: "markdown",
  metadata: {},
  source: src.split("\n").map((l, i, a) => (i === a.length - 1 ? l : l + "\n")),
});

const intro = `# 📈 Abhinav's Portfolio Manager — Live Web App

This notebook **is the backend**. Press **Runtime → Run all** and you'll get a public website URL.

- Tab 1: Live portfolio valuation (your ISA)
- Tab 2: One ranked **conviction buy list**
- Tab 3: Markowitz efficient frontier over your ETFs **+ candidate stocks**
- Tab 4: Value screener

> The URL works while this Colab tab stays open (Colab idles out after ~90 min). To make it permanent later, deploy the same \`app.py\` to Streamlit Community Cloud.`;

const install = `# ── CELL 1: Install (pinned) — ~90s ───────────────────────────────────────────
!pip install -q "yfinance==0.2.51" "streamlit==1.39.0" "PyPortfolioOpt==1.5.5" plotly
# Cloudflare tunnel binary (no signup, no password page)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared
print("✅ Dependencies installed")`;

const writePriv = priv ? `%%writefile _private_holdings.py\n${priv}` : null;
const writeApp = `%%writefile app.py\n${app}`;

const launch = `# ── CELL 3: Launch the web app + public URL ───────────────────────────────────
import subprocess, time, re, threading, os, signal

# kill any previous run
os.system("pkill -f streamlit; pkill -f cloudflared")
time.sleep(2)

# start streamlit (headless) in the background.
# CORS/XSRF/compression flags fix the "Bad message format / SessionInfo" error behind a tunnel.
streamlit = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501",
     "--server.headless", "true", "--server.address", "0.0.0.0",
     "--server.enableCORS", "false",
     "--server.enableXsrfProtection", "false",
     "--server.enableWebsocketCompression", "false",
     "--browser.gatherUsageStats", "false"],
    stdout=open("st.log", "w"), stderr=subprocess.STDOUT, text=True,
)
print("⏳ Starting Streamlit…")
time.sleep(8)

# open the Cloudflare tunnel and capture the public URL
tunnel = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8501"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)

url = None
start = time.time()
for line in tunnel.stdout:
    m = re.search(r"https://[-a-z0-9]+\\.trycloudflare\\.com", line)
    if m:
        url = m.group(0)
        break
    if time.time() - start > 40:
        break

if url:
    print("\\n" + "=" * 60)
    print("✅ YOUR LIVE WEBSITE IS READY — click this link:")
    print(f"   {url}")
    print("=" * 60)
    print("\\n(Keep this Colab tab open. The link dies when Colab stops.)")
else:
    print("⚠️ Could not detect the tunnel URL. Re-run this cell.")`;

const nb = {
  nbformat: 4,
  nbformat_minor: 5,
  metadata: {
    kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
    language_info: { name: "python" },
    colab: { provenance: [] },
  },
  cells: [
    mdCell(intro),
    codeCell(install),
    ...(writePriv ? [codeCell(writePriv)] : []),
    codeCell(writeApp),
    codeCell(launch),
  ],
};

writeFileSync(new URL("./portfolio_app_launcher.ipynb", import.meta.url), JSON.stringify(nb, null, 1));
console.log("✅ Wrote portfolio_app_launcher.ipynb");
