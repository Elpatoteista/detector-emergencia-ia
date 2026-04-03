import {
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

// ─────────────────────────────────────────────────────────────────────────────
// CONFIGURACIÓN (Mismos umbrales que la versión Python)
// ─────────────────────────────────────────────────────────────────────────────
const EAR_THRESHOLD = 0.21;
const EYE_ASYMMETRY_SECONDS = 5;
const BOTH_EYES_CLOSED_SECONDS = 5;
const MOUTH_ASYMMETRY_THRESHOLD = 0.15;
const MOUTH_ASYMMETRY_SECONDS = 5;
const HEAD_TILT_ANGLE = 45;
const HEAD_TILT_SECONDS = 3;
const NO_BLINK_WARNING_SECONDS = 8; // Actualizado a petición

// ── DOM Elements ──
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const loader = document.getElementById("loader");
const loadStatus = document.getElementById("load-status");
const emergencyOverlay = document.getElementById("emergency-overlay");
const alertList = document.getElementById("alert-list");
const btnRecalibrate = document.getElementById("btn-recalibrate");
const beepSound = document.getElementById("beep-sound");
const volumeSlider = document.getElementById("volume-slider");
const volumeValue = document.getElementById("volume-value");

// Inicializar volumen
beepSound.volume = 0.5;

// ── Estado de la Aplicación ──
let faceLandmarker;
let runningMode = "VIDEO";
let lastVideoTime = -1;

// Temporizadores
let timers = {
    eyeAsym: null,
    eyesClosed: null,
    mouthAsym: null,
    headTilt: null,
    lastBlink: Date.now()
};

// Emergencias activas
let alerts = {
    eyeAsym: false,
    eyesClosed: false,
    mouthAsym: false,
    headTilt: false,
    noBlink: false
};

// Calibración
let baselineHeadAngle = null;
let calibrationSamples = [];
const CALIBRATION_FRAMES_NEEDED = 60;
let framesCalibrated = 0;

// Parpadeo
let blinkCount = 0;
let prevEyesClosed = false;
let bpm = 0;
let blinkStartTime = Date.now();

// ─────────────────────────────────────────────────────────────────────────────
// FUNCIONES MATEMÁTICAS
// ─────────────────────────────────────────────────────────────────────────────
function euclidean(p1, p2) {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
}

function calculateEAR(landmarks, indices) {
    const p1 = landmarks[indices[0]];
    const p2 = landmarks[indices[1]];
    const p3 = landmarks[indices[2]];
    const p4 = landmarks[indices[3]];
    const p5 = landmarks[indices[4]];
    const p6 = landmarks[indices[5]];

    const v1 = euclidean(p2, p6);
    const v2 = euclidean(p3, p5);
    const h = euclidean(p1, p4);
    
    return h === 0 ? 0 : (v1 + v2) / (2.0 * h);
}

function calculateHeadTilt(landmarks) {
    // Landmark 33 (ojo izq externo) y 263 (ojo der externo)
    const p1 = landmarks[33];
    const p2 = landmarks[263];
    const dy = p2.y - p1.y;
    const dx = p2.x - p1.x;
    return Math.atan2(dy, dx) * (180 / Math.PI);
}

function calculateMouthAsymmetry(landmarks) {
    const nose = landmarks[1];
    const leftCorner = landmarks[61];
    const rightCorner = landmarks[291];

    const distLeft = leftCorner.y - nose.y;
    const distRight = rightCorner.y - nose.y;

    const diff = Math.abs(distLeft - distRight);
    const avg = (Math.abs(distLeft) + Math.abs(distRight)) / 2.0;
    
    if (avg === 0) return 0;
    return diff / avg;
}

// ─────────────────────────────────────────────────────────────────────────────
// MOTOR PRINCIPAL
// ─────────────────────────────────────────────────────────────────────────────
async function initialize() {
    try {
        loadStatus.innerText = "Cargando modelos de visión...";
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        
        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU"
            },
            outputFaceBlendshapes: false,
            runningMode: runningMode,
            numFaces: 1
        });

        loadStatus.innerText = "Accediendo a la cámara...";
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 }
        });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            loader.classList.add("hidden");
            predictWebcam();
        });

    } catch (error) {
        console.error(error);
        alert("Error al iniciar la cámara o IA. Asegúrate de dar permisos.");
    }
}

async function predictWebcam() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        const results = faceLandmarker.detectForVideo(video, Date.now());
        processResults(results);
    }
    
    window.requestAnimationFrame(predictWebcam);
}

function processResults(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];
        
        // 1. OJOS (EAR)
        const earLeft = calculateEAR(landmarks, [33, 160, 158, 133, 153, 144]);
        const earRight = calculateEAR(landmarks, [362, 385, 387, 263, 373, 380]);
        updateEyesLogic(earLeft, earRight);

        // 2. BOCA
        const mouthAsym = calculateMouthAsymmetry(landmarks);
        updateMouthLogic(mouthAsym);

        // 3. CABEZA
        const headTilt = calculateHeadTilt(landmarks);
        updateHeadLogic(headTilt);

        // 4. PARPADEO
        updateBlinkLogic(earLeft, earRight);
        
        // DIBUJAR (opcional, solo puntos clave para estética)
        drawFaceOverlay(landmarks);
    } else {
        resetTimers();
    }

    updateEmergencyUI();
}

// ─────────────────────────────────────────────────────────────────────────────
// LÓGICA DE DETECCIÓN Y TIEMPOS
// ─────────────────────────────────────────────────────────────────────────────
function updateEyesLogic(l, r) {
    const leftOpen = l > EAR_THRESHOLD;
    const rightOpen = r > EAR_THRESHOLD;
    
    document.getElementById("ear-left").innerText = l.toFixed(2);
    document.getElementById("ear-right").innerText = r.toFixed(2);

    if (leftOpen && rightOpen) {
        timers.eyeAsym = null;
        timers.eyesClosed = null;
        alerts.eyeAsym = false;
        alerts.eyesClosed = false;
        setCardStatus("card-eyes", "Normal", "bar-eyes", 0);
    } else if (!leftOpen && !rightOpen) {
        timers.eyeAsym = null;
        if (!timers.eyesClosed) timers.eyesClosed = Date.now();
        const elapsed = (Date.now() - timers.eyesClosed) / 1000;
        setCardStatus("card-eyes", "Ojos Cerrados", "bar-eyes", elapsed / BOTH_EYES_CLOSED_SECONDS);
        if (elapsed >= BOTH_EYES_CLOSED_SECONDS) alerts.eyesClosed = true;
    } else {
        timers.eyesClosed = null;
        if (!timers.eyeAsym) timers.eyeAsym = Date.now();
        const elapsed = (Date.now() - timers.eyeAsym) / 1000;
        setCardStatus("card-eyes", "Asimetría Ocular", "bar-eyes", elapsed / EYE_ASYMMETRY_SECONDS);
        if (elapsed >= EYE_ASYMMETRY_SECONDS) alerts.eyeAsym = true;
    }
}

function updateMouthLogic(asym) {
    const perc = (asym * 100).toFixed(1);
    document.getElementById("mouth-asym").innerText = `${perc}%`;

    if (asym >= MOUTH_ASYMMETRY_THRESHOLD) {
        if (!timers.mouthAsym) timers.mouthAsym = Date.now();
        const elapsed = (Date.now() - timers.mouthAsym) / 1000;
        setCardStatus("card-mouth", "Posible Parálisis", "bar-mouth", elapsed / MOUTH_ASYMMETRY_SECONDS);
        if (elapsed >= MOUTH_ASYMMETRY_SECONDS) alerts.mouthAsym = true;
    } else {
        timers.mouthAsym = null;
        alerts.mouthAsym = false;
        setCardStatus("card-mouth", "Simétrica", "bar-mouth", 0);
    }
}

function updateHeadLogic(angle) {
    if (framesCalibrated < CALIBRATION_FRAMES_NEEDED) {
        calibrationSamples.push(angle);
        framesCalibrated++;
        if (framesCalibrated === CALIBRATION_FRAMES_NEEDED) {
            baselineHeadAngle = calibrationSamples.reduce((a,b) => a+b) / calibrationSamples.length;
        }
        document.getElementById("detail-head").innerText = "Calibrando...";
        return;
    }

    const diff = Math.abs(angle - baselineHeadAngle);
    document.getElementById("head-tilt").innerText = `${Math.round(diff)}°`;

    if (diff >= HEAD_TILT_ANGLE) {
        if (!timers.headTilt) timers.headTilt = Date.now();
        const elapsed = (Date.now() - timers.headTilt) / 1000;
        setCardStatus("card-head", "Caída detectada", "bar-head", elapsed / HEAD_TILT_SECONDS);
        if (elapsed >= HEAD_TILT_SECONDS) alerts.headTilt = true;
    } else {
        timers.headTilt = null;
        alerts.headTilt = false;
        setCardStatus("card-head", "Estable", "bar-head", 0);
    }
}

function updateBlinkLogic(l, r) {
    const closed = (l <= EAR_THRESHOLD && r <= EAR_THRESHOLD);
    
    if (prevEyesClosed && !closed) {
        blinkCount++;
        timers.lastBlink = Date.now();
    }
    prevEyesClosed = closed;

    const noBlinkTime = (Date.now() - timers.lastBlink) / 1000;
    document.getElementById("last-blink").innerText = `${Math.round(noBlinkTime)}s`;
    
    // BPM simple (basado en el minuto actual)
    const elapsedTotal = (Date.now() - blinkStartTime) / 1000;
    bpm = Math.round(blinkCount * (60 / Math.max(elapsedTotal, 1)));
    document.getElementById("bpm").innerText = bpm;

    if (noBlinkTime >= NO_BLINK_WARNING_SECONDS) {
        alerts.noBlink = true;
        setCardStatus("card-blink", "Ausencia prolongada", "bar-blink", 1);
    } else {
        alerts.noBlink = false;
        setCardStatus("card-blink", "Normal", "bar-blink", noBlinkTime / NO_BLINK_WARNING_SECONDS);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// UI HELPERS
// ─────────────────────────────────────────────────────────────────────────────
function setCardStatus(cardId, detail, barId, progress) {
    const card = document.getElementById(cardId);
    const detailEl = document.getElementById(`detail-${cardId.split('-')[1]}`);
    const bar = document.getElementById(barId);
    
    detailEl.innerText = detail;
    bar.style.width = `${Math.min(progress * 100, 100)}%`;
    
    if (progress > 0) {
        card.classList.add("active");
    } else {
        card.classList.remove("active");
    }
}

function updateEmergencyUI() {
    const activeAlerts = [];
    if (alerts.eyeAsym) activeAlerts.push("Asimetría Ocular Crítica");
    if (alerts.eyesClosed) activeAlerts.push("Ojos Cerrados (Pérdida Conciencia)");
    if (alerts.mouthAsym) activeAlerts.push("Parálisis Facial Detectada");
    if (alerts.headTilt) activeAlerts.push("Inclinación de Cabeza Brusca");
    if (alerts.noBlink) activeAlerts.push("Falta de Parpadeo Crítica");

    if (activeAlerts.length > 0) {
        emergencyOverlay.classList.remove("hidden");
        alertList.innerHTML = activeAlerts.map(a => `<li>${a}</li>`).join("");
        if (beepSound.paused) beepSound.play().catch(() => {});
    } else {
        emergencyOverlay.classList.add("hidden");
    }
}

function drawFaceOverlay(landmarks) {
    ctx.strokeStyle = "rgba(51, 255, 153, 0.5)";
    ctx.lineWidth = 1;

    // Solo dibujamos los contornos de ojos y boca para estética premium
    const drawIndices = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 61, 291];
    drawIndices.forEach(i => {
        const pt = landmarks[i];
        ctx.beginPath();
        ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 2, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(51, 255, 153, 0.8)";
        ctx.fill();
    });
}

function resetTimers() {
    timers.eyeAsym = null;
    timers.mouthAsym = null;
    timers.headTilt = null;
}

btnRecalibrate.onclick = () => {
    framesCalibrated = 0;
    calibrationSamples = [];
    blinkCount = 0;
    blinkStartTime = Date.now();
    timers.lastBlink = Date.now();
    Object.keys(alerts).forEach(k => alerts[k] = false);
};

volumeSlider.oninput = (e) => {
    const val = e.target.value;
    beepSound.volume = val;
    volumeValue.innerText = `${Math.round(val * 100)}%`;
};

initialize();
