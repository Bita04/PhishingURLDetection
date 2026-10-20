# Explainable Phishing URL Detector

A small Python machine-learning project with a FastAPI backend and one React/Vite
frontend. The frontend runs as a local URL-checking page or as a Chrome
Manifest V3 popup for the current tab.

> **This tool gives an estimate, not a guarantee.** It does not block pages or
> replace browser warnings and security tools.

## How it works

In beginner terms: the tool looks at the hostname's spelling (for example,
digits, hyphens, and subdomains) and whether the URL uses HTTPS, then compares
those measurements with patterns learned from labeled phishing and legitimate
URLs. It returns an estimated phishing probability and uses SHAP to describe
the three strongest signals for that specific prediction. It never opens the
URL, looks at a web page, or contacts the destination.

## Run it locally

Use Python 3.10 or newer and Node.js 20 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the UCI archive to `backend/data/phiusiil.zip` if it is not already
there:

```bash
mkdir -p backend/data
curl -L --fail 'https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip' -o backend/data/phiusiil.zip
```

Train all models, evaluate both splits, save the app model and create the SHAP
plot:

```bash
python backend/train.py
```

Start the API in a terminal and leave it running:

```bash
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

For the web demo, in another terminal:

```bash
cd extension
npm install
npm run dev
```

Open the local address printed by Vite (normally
<http://127.0.0.1:5173>). Paste a URL and select **Check**.

### Chrome extension

With the API running, build and load the same app as an extension:

```bash
cd extension
npm install
npm run build
```

In Chrome, visit `chrome://extensions`, enable **Developer mode**, choose
**Load unpacked**, and select `extension/dist`. Open an HTTP(S) website and
click the extension icon. The extension uses `activeTab` and connects only to
the local API at `127.0.0.1:8000`.

## Dataset and limitations

Training uses the [UCI PhiUSIIL Phishing URL (Website) Dataset][dataset] under
CC BY 4.0. It contains 235,795 URLs: 100,945 phishing examples (42.8%) and
134,850 legitimate examples (57.2%). The target is `0 = phishing`, `1 =
legitimate`. The data was collected for a 2024 study and last updated by UCI
in May 2024. The downloaded archive is ignored by Git; training reads `URL`
and `label` only. No precomputed or webpage-derived features are used.

Important limitations:

- The dataset is now dated; URLs, attackers, and legitimate sites change
  quickly. Its class balance and source collection are not the same as live
  browser traffic.
- The source collection has artifacts: every legitimate source URL has a
  `www` hostname, and phishing examples disproportionately include paths.
  Feature extraction removes a leading `www` and excludes URL paths and query
  strings so models cannot rely on those collection shortcuts.
- The model therefore cannot detect a malicious path hosted on an otherwise
  familiar domain, and it does not inspect path or query text. Lexical
  hostname patterns can also be changed or evaded; a convincing phishing
  hostname can look ordinary, and a legitimate hostname can look unusual.
- HTTPS does not prove legitimacy. It is one learned signal, not a security
  guarantee. The model cannot see page content, redirects, domain history,
  certificates, or reputation.
- The hostname-held-out split prevents the exact same hostname from appearing
  in both train and test, but is not a time-based or registrable-domain split.
- Scores are not calibrated to the prevalence of phishing on the public web.
  A "Safe signals" verdict is not evidence that a site is trustworthy.

The trained model and evaluation artifacts are reproducible with seed 42. All
three models use the same normalized, hostname-only feature set. Logistic
Regression is the user-facing model because its behavior is simple to explain;
the model reports do not imply that it is ready for production. Manual
regression checks include bare and `www` forms of popular sites and realistic
paths such as Gmail inbox and GitHub login. Paths do not affect this model's
score. These are sanity checks, not a representative real-world benchmark.

## Results

Metrics below were produced by `python backend/train.py` on the downloaded UCI
archive. Phishing (dataset label 0) is the positive class; the confusion
matrix columns are TN / FP / FN / TP. Random split is stratified 80/20.
Hostname-held-out uses `GroupShuffleSplit` (80/20) and had zero hostname
overlap after normalizing away `www`: 188,840 train rows and 46,955 test rows.

| Model | Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion matrix (TN/FP/FN/TP) |
|---|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | Stratified random | 0.9081 | 0.9475 | 0.8314 | 0.8857 | 0.9309 | 26,040 / 930 / 3,404 / 16,785 |
| Random Forest | Stratified random | 0.9145 | 0.9366 | 0.8583 | 0.8958 | 0.9552 | 25,797 / 1,173 / 2,860 / 17,329 |
| LightGBM | Stratified random | 0.9176 | 0.9620 | 0.8409 | 0.8973 | 0.9552 | 26,299 / 671 / 3,213 / 16,976 |
| Logistic Regression | Hostname-held-out | 0.9028 | 0.9470 | 0.8164 | 0.8769 | 0.9406 | 26,143 / 909 / 3,654 / 16,249 |
| Random Forest | Hostname-held-out | 0.9062 | 0.9312 | 0.8408 | 0.8837 | 0.9474 | 25,815 / 1,237 / 3,168 / 16,735 |
| LightGBM | Hostname-held-out | 0.9061 | 0.9310 | 0.8408 | 0.8836 | 0.9475 | 25,811 / 1,241 / 3,168 / 16,735 |

Full-precision rows and split metadata are saved in
[`backend/results/metrics.csv`](backend/results/metrics.csv) and
[`backend/results/summary.json`](backend/results/summary.json). The global
SHAP plot is [`backend/results/shap_summary.png`](backend/results/shap_summary.png).

## Screenshots

Web demo: _screenshot placeholder — add a browser screenshot here after a local run._

Chrome popup: _screenshot placeholder — add a Chrome screenshot here after loading the unpacked extension._

## API

The local server accepts `POST /predict` with a JSON body such as
`{"url":"https://example.com/path"}`. It returns a label (`safe` or
`suspicious`), estimated phishing probability, and three readable reasons.
Non-HTTP(S), malformed, or overlong inputs are rejected. `/health` is available
for checking that the server is running.

## Ethics, privacy, and future work

URLs entered into the page or checked by the extension are sent only to the
local FastAPI server on the same machine. This project does not log or persist
submitted URLs and does not make requests to the destination. Avoid checking
URLs containing private session tokens. The tool informs but never blocks
navigation; false positives and false negatives remain possible.

Next steps: validate on a newer time-stamped dataset, group by registrable
domain, calibrate thresholds against realistic traffic, test across languages
and top-level domains, and perform user studies before considering any
protective action.

[dataset]: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
