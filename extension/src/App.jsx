import React, { useEffect, useState } from "react";
import "./styles.css";

const API_URL = "http://127.0.0.1:8000/predict";

function UrlInput({ value, onChange, onSubmit, disabled }) {
  return (
    <form className="url-form" onSubmit={onSubmit}>
      <label htmlFor="url">Website URL</label>
      <div className="input-row">
        <input
          id="url"
          type="url"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder="https://example.com"
          autoComplete="url"
          maxLength={4096}
          required
          disabled={disabled}
        />
        <button type="submit" disabled={disabled || !value.trim()}>
          Check
        </button>
      </div>
    </form>
  );
}

function ReasonList({ reasons }) {
  return (
    <ul className="reason-list">
      {reasons.map((reason) => <li key={reason}>{reason}</li>)}
    </ul>
  );
}

function ResultCard({ result }) {
  const suspicious = result.label === "suspicious";
  const percent = Math.round(result.probability * 100);
  return (
    <section
      className={`result-card ${suspicious ? "is-suspicious" : "is-safe"}`}
      aria-labelledby="verdict"
    >
      <div className="verdict-row">
        <span className="verdict-icon" aria-hidden="true">{suspicious ? "!" : "✓"}</span>
        <div>
          <p className="eyebrow">URL CHECK</p>
          <h2 id="verdict">{suspicious ? "Suspicious" : "Safe signals"}</h2>
        </div>
      </div>
      <div
        className="probability"
        role="meter"
        aria-label="Estimated phishing probability"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={percent}
      >
        <div className="probability-label">
          <span>Estimated phishing probability</span>
          <strong>{percent}%</strong>
        </div>
        <div className="bar-track">
          <div className="bar-fill" style={{ width: `${percent}%` }} />
        </div>
      </div>
      <h3>Top model signals</h3>
      <ReasonList reasons={result.reasons} />
    </section>
  );
}

export default function App() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [extensionMode, setExtensionMode] = useState(false);

  useEffect(() => {
    const chromeApi = globalThis.chrome;
    if (!chromeApi?.tabs?.query || !chromeApi?.runtime?.id) return;

    setExtensionMode(true);
    setStatus("loading");
    chromeApi.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      if (chromeApi.runtime.lastError) {
        setError("Could not read the active tab. Try opening a regular website.");
        setStatus("error");
        return;
      }
      if (!tab?.url || !/^https?:\/\//i.test(tab.url)) {
        setError("Open an HTTP or HTTPS website, then open this extension again.");
        setStatus("error");
        return;
      }
      setUrl(tab.url);
      void checkUrl(tab.url);
    });
  }, []);

  async function checkUrl(value) {
    setStatus("loading");
    setError("");
    setResult(null);
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: value.trim() }),
      });
      const payload = await response.json();
      if (!response.ok) {
        const detail = Array.isArray(payload.detail)
          ? payload.detail.map((item) => item.msg).join(" ")
          : payload.detail;
        throw new Error(detail || "The URL could not be checked.");
      }
      setResult(payload);
      setStatus("ready");
    } catch (requestError) {
      if (requestError instanceof TypeError) {
        setError("The local server isn't running. Start the API, then try again.");
      } else {
        setError(requestError.message || "Something went wrong. Please try again.");
      }
      setStatus("error");
    }
  }

  function submit(event) {
    event.preventDefault();
    if (url.trim()) void checkUrl(url);
  }

  return (
    <main className={`app-shell ${extensionMode ? "extension-mode" : "web-mode"}`}>
      <header className="header">
        <img className="brand-icon" src="/icon.svg" alt="" />
        <div>
          <p className="eyebrow">PRIVATE, LOCAL CHECK</p>
          <h1>Phishing URL check</h1>
        </div>
      </header>
      <p className="intro">
        Review URL structure for signals linked with phishing. The website is never opened.
      </p>
      {!extensionMode && (
        <UrlInput value={url} onChange={setUrl} onSubmit={submit} disabled={status === "loading"} />
      )}
      {extensionMode && <p className="current-url" title={url}>{url}</p>}
      <p className="status" role="status" aria-live="polite">
        {status === "loading" && <><span className="spinner" aria-hidden="true" /> Checking URL…</>}
        {status === "error" && error}
        {status === "ready" && "Check complete."}
      </p>
      {status === "error" && !extensionMode && (
        <button className="retry-button" type="button" onClick={submit} disabled={!url.trim()}>
          Try again
        </button>
      )}
      {result && <ResultCard result={result} />}
      <footer>This tool gives an estimate, not a guarantee.</footer>
    </main>
  );
}
