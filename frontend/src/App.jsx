import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const SAMPLE_MESSAGES = [
  {
    id: "sample-1",
    sender: "Bank Alert",
    category: "spam",
    text: "Your bank account will be locked within 24 hours. Verify now at http://fake-bank-alert.com",
  },
  {
    id: "sample-2",
    sender: "Mom",
    category: "ham",
    text: "Please buy milk and bread on your way home tonight.",
  },
  {
    id: "sample-3",
    sender: "Brand SMS",
    category: "spam",
    text: "Congratulations, you won a brand new iPhone. Click the link below to claim your reward now.",
  },
  {
    id: "sample-4",
    sender: "Linh",
    category: "ham",
    text: "Meeting starts at 8:30 tomorrow morning. Please bring the slides.",
  },
];

const DEFAULT_MODEL = "SVM";

function formatClock(date = new Date()) {
  return date.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getLabelText(prediction) {
  if (!prediction) {
    return "--";
  }

  return prediction.label === "spam" ? "Fraud / Spam" : "Normal";
}

function getRiskText(prediction) {
  if (!prediction) {
    return "--";
  }

  const score = Math.round(prediction.confidence * 100);

  if (prediction.label === "spam") {
    if (score >= 85) {
      return "High";
    }
    if (score >= 65) {
      return "Medium";
    }
    return "Review";
  }

  if (score >= 85) {
    return "Low";
  }
  if (score >= 65) {
    return "Guarded";
  }
  return "Unclear";
}

function buildSignals(text, prediction) {
  if (!text.trim()) {
    return ["No message has been checked yet."];
  }

  const normalized = text.toLowerCase();
  const reasons = [];

  if (/http|www|bit\.ly|tinyurl|link/.test(normalized)) {
    reasons.push("Contains a suspicious link.");
  }
  if (/verify|password|otp|account|login|security/.test(normalized)) {
    reasons.push("Requests account verification or sensitive details.");
  }
  if (/won|winner|prize|reward|free|gift|congratulations/.test(normalized)) {
    reasons.push("Uses reward or prize wording.");
  }
  if (/urgent|immediately|within 24 hours|locked|suspended|action required/.test(normalized)) {
    reasons.push("Creates urgency or fear.");
  }

  if (!reasons.length && prediction?.label === "spam") {
    reasons.push("Matches common spam message patterns.");
  }

  if (!reasons.length) {
    reasons.push("No strong scam signal was found in this message.");
  }

  return reasons.slice(0, 4);
}

function App() {
  const [sender, setSender] = useState(SAMPLE_MESSAGES[0].sender);
  const [message, setMessage] = useState(SAMPLE_MESSAGES[0].text);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);
  const [availableModels, setAvailableModels] = useState([]);
  const [apiStatus, setApiStatus] = useState("checking");
  const [apiDetails, setApiDetails] = useState("Checking backend connection...");
  const [prediction, setPrediction] = useState(null);
  const [receivedMessage, setReceivedMessage] = useState(null);
  const [history, setHistory] = useState([]);
  const [isChecking, setIsChecking] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    let isMounted = true;

    async function bootstrap() {
      try {
        const [healthResponse, modelsResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/health`),
          fetch(`${API_BASE_URL}/models`),
        ]);

        if (!healthResponse.ok || !modelsResponse.ok) {
          throw new Error("Backend response is not valid.");
        }

        const healthData = await healthResponse.json();
        const modelsData = await modelsResponse.json();

        if (!isMounted) {
          return;
        }

        const loadedModels = Array.isArray(modelsData.models)
          ? modelsData.models.filter((item) => item.loaded)
          : [];

        const loadedNames = loadedModels.map((item) => item.name);

        setAvailableModels(modelsData.models || []);
        setApiStatus("online");
        setApiDetails(
          loadedNames.length
            ? `Ready: ${loadedNames.join(", ")}`
            : `Backend online but no models are loaded. Keys: ${(healthData.models_loaded || []).join(", ")}`,
        );

        if (loadedNames.length) {
          setSelectedModel((current) => {
            if (loadedNames.includes(current)) {
              return current;
            }
            return loadedNames.includes(DEFAULT_MODEL) ? DEFAULT_MODEL : loadedNames[0];
          });
        }
      } catch (error) {
        if (!isMounted) {
          return;
        }

        setApiStatus("offline");
        setApiDetails("Backend is offline. Run python -m uvicorn api:app --reload");
      }
    }

    bootstrap();

    return () => {
      isMounted = false;
    };
  }, []);

  const selectedModelMeta = useMemo(
    () => availableModels.find((item) => item.name === selectedModel),
    [availableModels, selectedModel],
  );

  const warningSignals = useMemo(
    () => buildSignals(receivedMessage?.text || "", prediction),
    [receivedMessage, prediction],
  );

  function handleUseSample(sample) {
    setSender(sample.sender);
    setMessage(sample.text);
    setErrorMessage("");
  }

  function handleSendSms() {
    const trimmedSender = sender.trim() || "Unknown sender";
    const trimmedMessage = message.trim();

    if (!trimmedMessage) {
      setErrorMessage("Please enter a message before sending.");
      return;
    }

    setReceivedMessage({
      sender: trimmedSender,
      text: trimmedMessage,
      sentAt: formatClock(),
    });
    setPrediction(null);
    setErrorMessage("");
  }

  async function handleCheckMessage() {
    if (!receivedMessage?.text) {
      setErrorMessage("Send a message to the right phone first.");
      return;
    }

    setErrorMessage("");
    setIsChecking(true);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: receivedMessage.text,
          model: selectedModel,
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(data.detail || "Unable to analyze this message.");
      }

      setPrediction(data);
      setHistory((current) => [
        {
          id: crypto.randomUUID(),
          sender: receivedMessage.sender,
          text: receivedMessage.text,
          label: data.label,
          confidence: data.confidence,
          model: data.model,
          sentAt: receivedMessage.sentAt,
        },
        ...current,
      ].slice(0, 6));
    } catch (error) {
      setPrediction(null);
      setErrorMessage(error.message || "Something went wrong while calling the API.");
    } finally {
      setIsChecking(false);
    }
  }

  const confidenceText = prediction ? `${Math.round(prediction.confidence * 100)}%` : "--";
  const resultTone = prediction?.label === "spam" ? "danger" : prediction ? "safe" : "idle";

  return (
    <main className="app-shell">
      <section className="phone-stage">
        <article className="phone-card">
          <div className="phone-frame">
            <div className="phone-notch" />
            <div className="phone-screen">
              <div className="phone-statusbar">
                <span>{formatClock()}</span>
                <span>SMS Sender</span>
              </div>

              <header className="phone-header">
                <div>
                  <p className="phone-label">Phone 1</p>
                  <h1>Send SMS</h1>
                </div>
              </header>

              <div className="phone-scroll">
                <section className="box">
                  <label className="field-label" htmlFor="sender-input">
                    Sender
                  </label>
                  <input
                    id="sender-input"
                    className="field-input"
                    value={sender}
                    onChange={(event) => setSender(event.target.value)}
                    placeholder="Bank Alert"
                  />
                </section>

                <section className="box">
                  <label className="field-label" htmlFor="message-input">
                    SMS content
                  </label>
                  <textarea
                    id="message-input"
                    className="field-textarea"
                    value={message}
                    onChange={(event) => setMessage(event.target.value)}
                    placeholder="Type or paste an SMS message..."
                    rows={7}
                  />
                </section>

                <button className="primary-button" onClick={handleSendSms} type="button">
                  Send to right phone
                </button>

                <section className="box">
                  <div className="box-head">
                    <span className="field-label">Sample messages</span>
                    <span className="helper-text">{SAMPLE_MESSAGES.length} items</span>
                  </div>

                  <div className="sample-list">
                    {SAMPLE_MESSAGES.map((sample) => (
                      <button
                        key={sample.id}
                        className={`sample-card ${sample.category}`}
                        onClick={() => handleUseSample(sample)}
                        type="button"
                      >
                        <strong>{sample.sender}</strong>
                        <p>{sample.text}</p>
                      </button>
                    ))}
                  </div>
                </section>

                <section className="box">
                  <div className="box-head">
                    <span className="field-label">Preview</span>
                  </div>
                  <div className="sms-bubble sender-bubble">
                    <strong>{sender || "Unknown sender"}</strong>
                    <p>{message || "Your SMS preview will appear here."}</p>
                  </div>
                </section>
              </div>
            </div>
          </div>
        </article>

        <article className="phone-card">
          <div className="phone-frame">
            <div className="phone-notch" />
            <div className="phone-screen">
              <div className="phone-statusbar">
                <span>{receivedMessage?.sentAt || formatClock()}</span>
                <span>SMS Shield</span>
              </div>

              <header className="phone-header">
                <div>
                  <p className="phone-label">Phone 2</p>
                  <h1>Check SMS</h1>
                </div>
              </header>

              <div className="phone-scroll">
                <section className="box">
                  <div className="box-head">
                    <span className="field-label">Latest message</span>
                    <span className={`status-pill ${apiStatus}`}>{apiStatus}</span>
                  </div>

                  <div className="sms-bubble receiver-bubble">
                    <strong>{receivedMessage?.sender || "No sender yet"}</strong>
                    <p>{receivedMessage?.text || "The message sent from phone 1 will appear here."}</p>
                  </div>
                </section>

                <section className="box">
                  <div className="box-head">
                    <span className="field-label">Model</span>
                    <span className="helper-text">{selectedModelMeta?.loaded ? "Ready" : "Unavailable"}</span>
                  </div>

                  <select
                    className="field-select"
                    value={selectedModel}
                    onChange={(event) => setSelectedModel(event.target.value)}
                  >
                    {availableModels.length ? (
                      availableModels
                        .filter((item) => item.loaded)
                        .map((item) => (
                          <option key={item.name} value={item.name}>
                            {item.name}
                          </option>
                        ))
                    ) : (
                      <option value={selectedModel}>{selectedModel}</option>
                    )}
                  </select>
                </section>

                <button
                  className="primary-button"
                  disabled={isChecking || apiStatus !== "online" || !receivedMessage}
                  onClick={handleCheckMessage}
                  type="button"
                >
                  {isChecking ? "Checking..." : "Check message"}
                </button>

                <section className={`box result-box ${resultTone}`}>
                  <div className="box-head">
                    <span className="field-label">Result</span>
                    <span className="helper-text">{prediction?.model || selectedModel}</span>
                  </div>

                  <div className="result-grid">
                    <div>
                      <span>Type</span>
                      <strong>{getLabelText(prediction)}</strong>
                    </div>
                    <div>
                      <span>Risk</span>
                      <strong>{getRiskText(prediction)}</strong>
                    </div>
                    <div>
                      <span>Confidence</span>
                      <strong>{confidenceText}</strong>
                    </div>
                    <div>
                      <span>Status</span>
                      <strong>{apiStatus === "online" ? "Online" : "Offline"}</strong>
                    </div>
                  </div>
                </section>

                <section className="box">
                  <div className="box-head">
                    <span className="field-label">Signals</span>
                  </div>
                  <ul className="reason-list">
                    {warningSignals.map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                </section>

                {errorMessage ? (
                  <section className="box error-box">
                    <div className="box-head">
                      <span className="field-label">Error</span>
                    </div>
                    <p className="box-text">{errorMessage}</p>
                  </section>
                ) : null}

                <section className="box">
                  <div className="box-head">
                    <span className="field-label">History</span>
                    <span className="helper-text">{history.length} items</span>
                  </div>

                  {history.length ? (
                    <div className="history-list">
                      {history.map((item) => (
                        <article className="history-card" key={item.id}>
                          <div className="history-row">
                            <strong>{item.sender}</strong>
                            <span>{item.sentAt}</span>
                          </div>
                          <p>{item.text}</p>
                          <div className="history-row">
                            <span>{item.model}</span>
                            <span>
                              {item.label} | {Math.round(item.confidence * 100)}%
                            </span>
                          </div>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <p className="box-text">No checked messages yet.</p>
                  )}
                </section>

                <section className="box">
                  <div className="box-head">
                    <span className="field-label">API status</span>
                  </div>
                  <p className="box-text">{apiDetails}</p>
                </section>
              </div>
            </div>
          </div>
        </article>
      </section>
    </main>
  );
}

export default App;
