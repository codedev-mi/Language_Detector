// =====================================================
// SIGN LANGUAGE DETECTION - FRONTEND
// =====================================================

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

const sentenceEl = document.getElementById("sentence");
const currentLetterEl = document.getElementById("currentLetter");
const confidenceEl = document.getElementById("confidence");
const statusEl = document.getElementById("status");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");

const canvasContext = canvas.getContext("2d");

let cameraStream = null;
let detectionRunning = false;
let requestInProgress = false;

let detectionInterval = null;


// =====================================================
// START CAMERA AND DETECTION
// =====================================================

startBtn.addEventListener("click", async () => {

    if (detectionRunning) {
        return;
    }

    try {

        statusEl.textContent = "Requesting camera permission...";

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: {
                    ideal: 640
                },
                height: {
                    ideal: 480
                },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = cameraStream;

        await video.play();

        const response = await fetch("/start", {
            method: "POST"
        });

        const data = await response.json();

        detectionRunning = true;

        statusEl.textContent = "Detection is running";

        startBtn.disabled = true;
        stopBtn.disabled = false;

        sentenceEl.textContent = "";
        currentLetterEl.textContent = "-";
        confidenceEl.textContent = "Confidence: -";

        // Capture frames every 300 milliseconds
        detectionInterval = setInterval(
            captureAndPredict,
            300
        );

        console.log(data.message);

    } catch (error) {

        console.error("Camera error:", error);

        statusEl.textContent =
            "Unable to access camera. Please allow camera permission.";

        alert(
            "Camera access failed. Please allow camera permission."
        );

    }

});


// =====================================================
// CAPTURE FRAME AND SEND TO FLASK
// =====================================================

async function captureAndPredict() {

    if (!detectionRunning || requestInProgress) {
        return;
    }

    if (
        video.readyState < 2 ||
        video.videoWidth === 0
    ) {
        return;
    }

    requestInProgress = true;

    try {

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        canvasContext.drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height
        );

        // Convert frame to JPEG Base64
        const imageData = canvas.toDataURL(
            "image/jpeg",
            0.7
        );

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                image: imageData
            })

        });

        const data = await response.json();

        if (data.status === "success") {

            currentLetterEl.textContent =
                data.current || "-";

            sentenceEl.textContent =
                data.sentence || "";

            confidenceEl.textContent =
                "Confidence: " +
                (data.confidence * 100).toFixed(2) +
                "%";

        } else if (data.status === "no_hand") {

            currentLetterEl.textContent = "-";

            confidenceEl.textContent =
                "Confidence: -";

        }

    } catch (error) {

        console.error("Prediction error:", error);

        statusEl.textContent =
            "Connection error while detecting gesture.";

    } finally {

        requestInProgress = false;

    }

}


// =====================================================
// STOP DETECTION
// =====================================================

stopBtn.addEventListener("click", async () => {

    await stopDetection();

});

async function stopDetection() {

    detectionRunning = false;

    if (detectionInterval) {

        clearInterval(detectionInterval);

        detectionInterval = null;

    }

    if (cameraStream) {

        cameraStream.getTracks().forEach(track => {
            track.stop();
        });

        cameraStream = null;

    }

    video.srcObject = null;

    try {

        await fetch("/stop", {
            method: "POST"
        });

    } catch (error) {

        console.error("Stop error:", error);

    }

    statusEl.textContent =
        "Detection stopped";

    startBtn.disabled = false;
    stopBtn.disabled = true;

    currentLetterEl.textContent = "-";

    confidenceEl.textContent =
        "Confidence: -";

}


// =====================================================
// CLEAR SENTENCE
// =====================================================

clearBtn.addEventListener("click", async () => {

    try {

        const response = await fetch("/clear", {
            method: "POST"
        });

        const data = await response.json();

        sentenceEl.textContent = "";

        currentLetterEl.textContent = "-";

        confidenceEl.textContent =
            "Confidence: -";

        console.log(data.message);

    } catch (error) {

        console.error("Clear error:", error);

    }

});


// =====================================================
// INITIAL STATE
// =====================================================

stopBtn.disabled = true;
