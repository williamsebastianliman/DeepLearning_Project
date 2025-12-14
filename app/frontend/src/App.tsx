import { useRef, useState, useEffect } from "react";

const API_URL = "http://127.0.0.1:8000/predict";
const DEBOUNCE_MS = 500;

// Replace this with your actual class list from class_list_z.json
const CLASS_LIST = [
  "angel",
  "apple",
  "bat",
  "book",
  "candle",
  "castle",
  "cat",
  "cup",
  "dog",
  "door",
  "fish",
  "flamingo",
  "hexagon",
  "lantern",
  "light bulb",
  "mailbox",
  "necklace",
  "octopus",
  "parachute",
  "pencil",
  "pig",
  "skull",
  "swan",
  "sword",
  "table",
  "van",
];

function App() {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const drawingRef = useRef(false);
  const debounceRef = useRef(null);

  const [currentTarget, setCurrentTarget] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [score, setScore] = useState(0);
  const [round, setRound] = useState(1);
  const [gameState, setGameState] = useState("playing");
  const [timer, setTimer] = useState(20);
  const [usedTargets, setUsedTargets] = useState([]);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [currentPrediction, setCurrentPrediction] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resize = () => {
      const maxW = Math.min(window.innerWidth - 40, 800);
      const maxH = Math.min(window.innerHeight - 280, 600);
      canvas.width = maxW;
      canvas.height = maxH;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#1a1a1a";
      ctx.lineWidth = 5;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctxRef.current = ctx;
    };

    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  useEffect(() => {
    pickNewTarget();
  }, []);

  useEffect(() => {
    if (gameState !== "playing") return;

    const interval = setInterval(() => {
      setTimer((prev) => {
        if (prev <= 1) {
          setGameState("failed");
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [gameState, round]);

  const pickNewTarget = () => {
    const available = CLASS_LIST.filter((c) => !usedTargets.includes(c));
    if (available.length === 0) {
      setUsedTargets([]);
      const target = CLASS_LIST[Math.floor(Math.random() * CLASS_LIST.length)];
      setCurrentTarget(target);
      return;
    }
    const target = available[Math.floor(Math.random() * available.length)];
    setCurrentTarget(target);
  };

  const getPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const startDraw = (e) => {
    drawingRef.current = true;
    const { x, y } = getPos(e);
    ctxRef.current.beginPath();
    ctxRef.current.moveTo(x, y);
  };

  const draw = (e) => {
    if (!drawingRef.current) return;
    const { x, y } = getPos(e);
    ctxRef.current.lineTo(x, y);
    ctxRef.current.stroke();
  };

  const endDraw = () => {
    drawingRef.current = false;
    ctxRef.current.closePath();

    if (debounceRef.current) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(sendToApi, DEBOUNCE_MS);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPredictions([]);
    setCurrentPrediction(null);
  };

  const sendToApi = async () => {
    const canvas = canvasRef.current;
    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/png")
    );
    if (!blob) return;

    const form = new FormData();
    form.append("file", blob, "drawing.png");

    try {
      const res = await fetch(API_URL, { method: "POST", body: form });
      const data = await res.json();

      // Store top 3 predictions
      const topPreds = data.top_predictions || [
        { class: data.predicted_class, confidence: data.confidence },
      ];
      setPredictions(topPreds.slice(0, 3));

      // Check if all top 3 are below 8% confidence
      const allLowConfidence = topPreds
        .slice(0, 3)
        .every((p) => p.confidence < 0.08);

      // Set current prediction display
      setCurrentPrediction({
        label: allLowConfidence ? "???" : data.predicted_class,
        confidence: data.confidence,
        isConfused: allLowConfidence,
      });

      // Check if target matches top 1, or top 2/3 if within 5% confidence threshold
      const top1 = topPreds[0];
      const top1Conf = top1.confidence;

      let isSuccess = top1.class.toLowerCase() === currentTarget.toLowerCase();

      if (!isSuccess && topPreds.length > 1) {
        // Check top 2 and 3 if they're within 5% of top 1
        for (let i = 1; i < Math.min(3, topPreds.length); i++) {
          if (Math.abs(topPreds[i].confidence - top1Conf) <= 0.05) {
            if (
              topPreds[i].class.toLowerCase() === currentTarget.toLowerCase()
            ) {
              isSuccess = true;
              break;
            }
          }
        }
      }

      if (isSuccess && gameState === "playing") {
        setScore((prev) => prev + Math.max(1, Math.floor(timer / 2)));
        setGameState("success");
        setShowSuccessModal(true);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const nextRound = () => {
    setShowSuccessModal(false);
    clearCanvas();
    setRound((prev) => prev + 1);
    setTimer(20);
    setGameState("playing");
    setUsedTargets((prev) => [...prev, currentTarget]);
    pickNewTarget();
  };

  const restartGame = () => {
    clearCanvas();
    setScore(0);
    setRound(1);
    setTimer(20);
    setGameState("playing");
    setUsedTargets([]);
    setShowSuccessModal(false);
    pickNewTarget();
  };

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        fontFamily: "'Poppins', -apple-system, system-ui, sans-serif",
        overflow: "hidden",
        padding: "20px",
      }}
    >
      {/* Header Stats */}
      <div
        style={{
          display: "flex",
          gap: "16px",
          marginBottom: "20px",
          flexWrap: "wrap",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            background: "#fff",
            padding: "16px 28px",
            borderRadius: "20px",
            fontWeight: 900,
            fontSize: "24px",
            boxShadow: "0 8px 24px rgba(0,0,0,0.2)",
            border: "4px solid #667eea",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            minWidth: "140px",
          }}
        >
          <div
            style={{
              fontSize: "14px",
              color: "#9ca3af",
              fontWeight: 600,
              marginBottom: "4px",
            }}
          >
            ROUND
          </div>
          <div style={{ fontSize: "42px", color: "#667eea", lineHeight: "1" }}>
            {round}
          </div>
        </div>

        <div
          style={{
            background: "#fff",
            padding: "16px 28px",
            borderRadius: "20px",
            fontWeight: 900,
            fontSize: "24px",
            boxShadow: "0 8px 24px rgba(0,0,0,0.2)",
            border: "4px solid #f59e0b",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            minWidth: "140px",
          }}
        >
          <div
            style={{
              fontSize: "14px",
              color: "#9ca3af",
              fontWeight: 600,
              marginBottom: "4px",
            }}
          >
            SCORE
          </div>
          <div style={{ fontSize: "42px", color: "#f59e0b", lineHeight: "1" }}>
            {score}
          </div>
        </div>

        <div
          style={{
            background: timer <= 5 ? "#ef4444" : "#fff",
            padding: "16px 28px",
            borderRadius: "20px",
            fontWeight: 900,
            fontSize: "24px",
            boxShadow: "0 8px 24px rgba(0,0,0,0.2)",
            border: timer <= 5 ? "4px solid #dc2626" : "4px solid #10b981",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            minWidth: "140px",
            transition: "all 0.3s ease",
            animation: timer <= 5 ? "pulse 1s ease-in-out infinite" : "none",
          }}
        >
          <div
            style={{
              fontSize: "14px",
              color: timer <= 5 ? "#fff" : "#9ca3af",
              fontWeight: 600,
              marginBottom: "4px",
            }}
          >
            TIME
          </div>
          <div
            style={{
              fontSize: "42px",
              color: timer <= 5 ? "#fff" : "#10b981",
              lineHeight: "1",
              display: "flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            <span style={{ fontSize: "32px" }}>⏱️</span>
            {timer}s
          </div>
        </div>
      </div>

      {/* Target Display */}
      <div
        style={{
          background: "rgba(255,255,255,0.98)",
          padding: "20px 40px",
          borderRadius: "20px",
          marginBottom: "20px",
          boxShadow: "0 12px 40px rgba(0,0,0,0.2)",
          textAlign: "center",
          maxWidth: "600px",
        }}
      >
        <div style={{ fontSize: "16px", color: "#666", marginBottom: "8px" }}>
          Draw this:
        </div>
        <div
          style={{
            fontSize: "36px",
            fontWeight: 800,
            color: "#667eea",
            textTransform: "uppercase",
            letterSpacing: "2px",
          }}
        >
          {currentTarget}
        </div>
      </div>

      {/* Canvas Container */}
      <div
        style={{
          position: "relative",
          background: "#fff",
          borderRadius: "20px",
          padding: "20px",
          boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
        }}
      >
        <canvas
          ref={canvasRef}
          onPointerDown={startDraw}
          onPointerMove={draw}
          onPointerUp={endDraw}
          onPointerLeave={endDraw}
          style={{
            border: "3px solid #e5e7eb",
            borderRadius: "12px",
            touchAction: "none",
            cursor: "crosshair",
            display: "block",
          }}
        />

        <button
          onClick={clearCanvas}
          style={{
            position: "absolute",
            bottom: "30px",
            right: "30px",
            padding: "10px 20px",
            borderRadius: "12px",
            border: "none",
            background: "#ef4444",
            color: "#fff",
            fontWeight: 700,
            fontSize: "14px",
            cursor: "pointer",
            boxShadow: "0 4px 12px rgba(239,68,68,0.3)",
            transition: "transform 0.2s",
          }}
          onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.95)")}
          onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
        >
          🗑️ Clear
        </button>
      </div>

      {currentPrediction && (
        <div
          style={{
            position: "fixed",
            top: "100px",
            right: "40px",
            zIndex: 100,
          }}
        >
          <div
            style={{
              position: "absolute",
              top: "-15px",
              right: "60px",
              display: "flex",
              gap: "8px",
            }}
          >
            <div
              style={{
                width: "12px",
                height: "12px",
                borderRadius: "50%",
                background: "#fff",
                border: "3px solid #667eea",
                animation: "float 2s ease-in-out infinite",
              }}
            />
            <div
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                background: "#fff",
                border: "3px solid #667eea",
                animation: "float 2s ease-in-out infinite 0.3s",
              }}
            />
          </div>

          <div
            style={{
              background: "#fff",
              padding: "24px 28px",
              borderRadius: "28px",
              boxShadow: "0 12px 40px rgba(0,0,0,0.2)",
              maxWidth: "320px",
              animation:
                "slideInBounce 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55)",
              border: "4px solid #667eea",
              position: "relative",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "10px",
                marginBottom: "12px",
              }}
            >
              <span style={{ fontSize: "32px" }}>🤖</span>
              <span
                style={{
                  fontSize: "16px",
                  color: "#667eea",
                  fontWeight: 800,
                  textTransform: "uppercase",
                  letterSpacing: "1px",
                }}
              >
                AI Detective
              </span>
            </div>

            <div
              style={{
                fontSize: "15px",
                color: "#6b7280",
                marginBottom: "12px",
                lineHeight: "1.5",
                fontWeight: 500,
              }}
            >
              {currentPrediction.isConfused
                ? "Umm... I have no idea what you're drawing! 🤔"
                : currentPrediction.confidence > 0.8
                ? "I'm pretty sure this is..."
                : currentPrediction.confidence > 0.5
                ? "Hmm, looks like..."
                : "Wild guess, but maybe..."}
            </div>

            {!currentPrediction.isConfused ? (
              <>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "12px",
                    flexWrap: "wrap",
                    marginBottom: "8px",
                  }}
                >
                  <div
                    style={{
                      fontSize: "28px",
                      fontWeight: 900,
                      color:
                        currentPrediction.label.toLowerCase() ===
                        currentTarget.toLowerCase()
                          ? "#10b981"
                          : "#667eea",
                      textTransform: "uppercase",
                      letterSpacing: "1px",
                    }}
                  >
                    {currentPrediction.label}
                  </div>
                  <div
                    style={{
                      fontSize: "14px",
                      color: "#fff",
                      background:
                        currentPrediction.confidence > 0.8
                          ? "#10b981"
                          : "#667eea",
                      padding: "6px 14px",
                      borderRadius: "12px",
                      fontWeight: 800,
                      boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
                    }}
                  >
                    {(currentPrediction.confidence * 100).toFixed(0)}%
                  </div>
                </div>

                {currentPrediction.label.toLowerCase() ===
                  currentTarget.toLowerCase() && (
                  <div
                    style={{
                      fontSize: "15px",
                      color: "#10b981",
                      marginTop: "8px",
                      fontWeight: 700,
                      display: "flex",
                      alignItems: "center",
                      gap: "6px",
                    }}
                  >
                    <span style={{ fontSize: "18px" }}>✨</span>
                    You're nailing it!
                  </div>
                )}
              </>
            ) : (
              <div
                style={{
                  fontSize: "24px",
                  fontWeight: 900,
                  color: "#ef4444",
                  textAlign: "center",
                  padding: "12px 0",
                }}
              >
                ¯\_(ツ)_/¯
              </div>
            )}
          </div>
        </div>
      )}

      {showSuccessModal && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.7)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
            animation: "fadeIn 0.3s ease",
          }}
        >
          <div
            style={{
              background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
              padding: "60px 80px",
              borderRadius: "30px",
              textAlign: "center",
              color: "#fff",
              boxShadow: "0 25px 80px rgba(0,0,0,0.5)",
              animation: "bounceIn 0.5s ease",
              maxWidth: "500px",
            }}
          >
            <div style={{ fontSize: "80px", marginBottom: "20px" }}>🎉</div>
            <div
              style={{
                fontSize: "40px",
                fontWeight: 900,
                marginBottom: "10px",
              }}
            >
              AMAZING!
            </div>
            <div
              style={{ fontSize: "20px", opacity: 0.9, marginBottom: "30px" }}
            >
              You drew <span style={{ fontWeight: 800 }}>{currentTarget}</span>{" "}
              perfectly!
            </div>
            <div style={{ fontSize: "24px", marginBottom: "40px" }}>
              +{Math.max(1, Math.floor(timer / 2))} points
            </div>
            <button
              onClick={nextRound}
              style={{
                padding: "16px 48px",
                borderRadius: "16px",
                border: "none",
                background: "#fff",
                color: "#10b981",
                fontSize: "20px",
                fontWeight: 800,
                cursor: "pointer",
                boxShadow: "0 8px 24px rgba(0,0,0,0.2)",
                transition: "transform 0.2s",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.transform = "scale(1.05)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.transform = "scale(1)")
              }
            >
              Next Round →
            </button>
          </div>
        </div>
      )}

      {gameState === "failed" && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.8)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
          }}
        >
          <div
            style={{
              background: "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)",
              padding: "60px 80px",
              borderRadius: "30px",
              textAlign: "center",
              color: "#fff",
              boxShadow: "0 25px 80px rgba(0,0,0,0.5)",
              maxWidth: "500px",
            }}
          >
            <div style={{ fontSize: "80px", marginBottom: "20px" }}>⏰</div>
            <div
              style={{
                fontSize: "40px",
                fontWeight: 900,
                marginBottom: "10px",
              }}
            >
              Time's Up!
            </div>
            <div
              style={{ fontSize: "20px", opacity: 0.9, marginBottom: "20px" }}
            >
              You needed to draw:{" "}
              <span style={{ fontWeight: 800 }}>{currentTarget}</span>
            </div>
            <div style={{ fontSize: "28px", marginBottom: "40px" }}>
              Final Score: <span style={{ fontWeight: 900 }}>{score}</span>
            </div>
            <button
              onClick={restartGame}
              style={{
                padding: "16px 48px",
                borderRadius: "16px",
                border: "none",
                background: "#fff",
                color: "#ef4444",
                fontSize: "20px",
                fontWeight: 800,
                cursor: "pointer",
                boxShadow: "0 8px 24px rgba(0,0,0,0.2)",
              }}
            >
              Play Again
            </button>
          </div>
        </div>
      )}

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes bounceIn {
          0% { transform: scale(0.3); opacity: 0; }
          50% { transform: scale(1.05); }
          70% { transform: scale(0.9); }
          100% { transform: scale(1); opacity: 1; }
        }
        @keyframes slideInBounce {
          0% { 
            transform: translateX(100px) scale(0.8);
            opacity: 0;
          }
          60% {
            transform: translateX(-10px) scale(1.05);
          }
          100% { 
            transform: translateX(0) scale(1);
            opacity: 1;
          }
        }
        @keyframes float {
          0%, 100% {
            transform: translateY(0px);
          }
          50% {
            transform: translateY(-8px);
          }
        }
        @keyframes pulse {
          0%, 100% {
            transform: scale(1);
          }
          50% {
            transform: scale(1.05);
          }
        }
      `}</style>
    </div>
  );
}

export default App;
