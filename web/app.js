/**
 * Digit Duel - Client-Side ONNX Runtime Web Inference Engine
 * Features:
 *   - Real-time client-side inference (<5ms)
 *   - Yann LeCun's center-of-mass centering algorithm
 *   - CNN vs RNN Duel Mode
 *   - Responsive touch & mouse drawing canvas
 *   - Image uploader support
 */

// Application State
const state = {
  activeModel: "cnn", // "cnn", "rnn", "duel"
  cnnSession: null,
  rnnSession: null,
  isDrawing: false,
  strokes: [],
  currentStroke: [],
  brushSize: 20,
};

// DOM Elements
const canvas = document.getElementById("paintCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const previewCanvas = document.getElementById("previewCanvas");
const previewCtx = previewCanvas.getContext("2d");
const canvasHint = document.getElementById("canvasHint");
const clearBtn = document.getElementById("clearBtn");
const undoBtn = document.getElementById("undoBtn");
const brushSizeSlider = document.getElementById("brushSize");
const latencyTag = document.getElementById("latencyTag");

// Tabs
const drawTabBtn = document.getElementById("drawTabBtn");
const uploadTabBtn = document.getElementById("uploadTabBtn");
const drawSection = document.getElementById("drawSection");
const uploadSection = document.getElementById("uploadSection");

// Upload
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const uploadPreviewWrapper = document.getElementById("uploadPreviewWrapper");
const uploadPreviewImg = document.getElementById("uploadPreviewImg");
const removeUploadBtn = document.getElementById("removeUploadBtn");

// Model Switcher
const modelToggleGroup = document.getElementById("modelToggleGroup");
const singleHero = document.getElementById("singleHero");
const duelContainer = document.getElementById("duelContainer");
const activeModelLabel = document.getElementById("activeModelLabel");
const topDigitEl = document.getElementById("topDigit");
const topConfidenceEl = document.getElementById("topConfidence");
const duelCnnDigit = document.getElementById("duelCnnDigit");
const duelCnnConf = document.getElementById("duelCnnConf");
const duelRnnDigit = document.getElementById("duelRnnDigit");
const duelRnnConf = document.getElementById("duelRnnConf");
const barsWrapper = document.getElementById("barsWrapper");

// Initialize Bar Chart DOM
function initBarChart() {
  barsWrapper.innerHTML = "";
  for (let i = 0; i <= 9; i++) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.id = `bar-row-${i}`;
    row.innerHTML = `
      <span class="bar-digit">${i}</span>
      <div class="bar-track">
        <div class="bar-fill" id="bar-fill-${i}"></div>
      </div>
      <span class="bar-pct" id="bar-pct-${i}">0.0%</span>
    `;
    barsWrapper.appendChild(row);
  }
}

// Initialize ONNX Sessions
async function initONNX() {
  latencyTag.textContent = "⏳ Loading Models...";
  try {
    // Configure ONNX Web
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    state.cnnSession = await ort.InferenceSession.create("models/cnn_mnist.onnx");
    console.log("[✓] CNN ONNX Session Loaded");

    try {
      state.rnnSession = await ort.InferenceSession.create("models/rnn_mnist.onnx");
      console.log("[✓] RNN ONNX Session Loaded");
    } catch (e) {
      console.warn("RNN ONNX not yet loaded or missing:", e);
    }

    latencyTag.textContent = "⚡ Ready (ONNX Web)";
  } catch (err) {
    console.error("Failed to load ONNX model:", err);
    latencyTag.textContent = "⚠️ Model Error";
  }
}

// Canvas Setup
function setupCanvas() {
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#FFFFFF";

  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let clientX = e.clientX;
    let clientY = e.clientY;

    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    }

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  }

  function startDrawing(e) {
    e.preventDefault();
    state.isDrawing = true;
    canvasHint.style.opacity = "0";

    const coords = getCanvasCoords(e);
    ctx.lineWidth = state.brushSize;
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
    state.currentStroke = [{ x: coords.x, y: coords.y, size: state.brushSize }];
  }

  function draw(e) {
    if (!state.isDrawing) return;
    e.preventDefault();

    const coords = getCanvasCoords(e);
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();

    state.currentStroke.push({ x: coords.x, y: coords.y, size: state.brushSize });
  }

  function stopDrawing(e) {
    if (!state.isDrawing) return;
    e.preventDefault();
    state.isDrawing = false;
    ctx.closePath();

    if (state.currentStroke.length > 0) {
      state.strokes.push([...state.currentStroke]);
      state.currentStroke = [];
    }

    triggerPrediction();
  }

  // Pointer & Touch Events
  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", draw);
  window.addEventListener("mouseup", stopDrawing);

  canvas.addEventListener("touchstart", startDrawing, { passive: false });
  canvas.addEventListener("touchmove", draw, { passive: false });
  canvas.addEventListener("touchend", stopDrawing, { passive: false });

  // Brush Slider
  brushSizeSlider.addEventListener("input", (e) => {
    state.brushSize = parseInt(e.target.value, 10);
  });

  // Clear Button
  clearBtn.addEventListener("click", () => {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    state.strokes = [];
    canvasHint.style.opacity = "1";
    resetOutput();
  });

  // Undo Button
  undoBtn.addEventListener("click", () => {
    if (state.strokes.length === 0) return;
    state.strokes.pop();
    redrawCanvas();
    if (state.strokes.length === 0) {
      canvasHint.style.opacity = "1";
      resetOutput();
    } else {
      triggerPrediction();
    }
  });
}

function redrawCanvas() {
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#FFFFFF";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const stroke of state.strokes) {
    if (stroke.length === 0) continue;
    ctx.lineWidth = stroke[0].size || 20;
    ctx.beginPath();
    ctx.moveTo(stroke[0].x, stroke[0].y);
    for (let i = 1; i < stroke.length; i++) {
      ctx.lineTo(stroke[i].x, stroke[i].y);
    }
    ctx.stroke();
  }
}

// Yann LeCun's MNIST Preprocessing: Center of Mass + Bounding Box Rescaling
function preprocessCanvas(sourceCanvas) {
  const srcCtx = sourceCanvas.getContext("2d", { willReadFrequently: true });
  const imgData = srcCtx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
  const data = imgData.data;

  // Convert to grayscale & extract bounding box
  let minX = sourceCanvas.width, maxX = 0;
  let minY = sourceCanvas.height, maxY = 0;
  let hasPixels = false;

  const grayGrid = new Float32Array(sourceCanvas.width * sourceCanvas.height);

  for (let y = 0; y < sourceCanvas.height; y++) {
    for (let x = 0; x < sourceCanvas.width; x++) {
      const idx = (y * sourceCanvas.width + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      // luminance
      let val = 0.299 * r + 0.587 * g + 0.114 * b;

      if (val > 25) { // threshold
        grayGrid[y * sourceCanvas.width + x] = val;
        hasPixels = true;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (!hasPixels) return null;

  // Crop to bounding box
  const boxW = Math.max(1, maxX - minX + 1);
  const boxH = Math.max(1, maxY - minY + 1);

  // Offscreen canvas for cropping
  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = boxW;
  cropCanvas.height = boxH;
  const cropCtx = cropCanvas.getContext("2d");
  cropCtx.drawImage(sourceCanvas, minX, minY, boxW, boxH, 0, 0, boxW, boxH);

  // Scale into 20x20 box preserving aspect ratio
  let targetW, targetH;
  if (boxW > boxH) {
    targetW = 20;
    targetH = Math.max(1, Math.round(boxH * (20 / boxW)));
  } else {
    targetH = 20;
    targetW = Math.max(1, Math.round(boxW * (20 / boxH)));
  }

  const scaledCanvas = document.createElement("canvas");
  scaledCanvas.width = targetW;
  scaledCanvas.height = targetH;
  const scaledCtx = scaledCanvas.getContext("2d");
  scaledCtx.drawImage(cropCanvas, 0, 0, targetW, targetH);

  const scaledData = scaledCtx.getImageData(0, 0, targetW, targetH).data;

  // Create 28x28 initial grid
  const grid28 = new Float32Array(28 * 28);
  const startX = Math.floor((28 - targetW) / 2);
  const startY = Math.floor((28 - targetH) / 2);

  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      const sIdx = (y * targetW + x) * 4;
      const val = 0.299 * scaledData[sIdx] + 0.587 * scaledData[sIdx + 1] + 0.114 * scaledData[sIdx + 2];
      grid28[(startY + y) * 28 + (startX + x)] = val;
    }
  }

  // Calculate Center of Mass
  let totalMass = 0;
  let sumX = 0;
  let sumY = 0;

  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const val = grid28[y * 28 + x];
      if (val > 0) {
        totalMass += val;
        sumX += x * val;
        sumY += y * val;
      }
    }
  }

  const final28 = new Float32Array(28 * 28);
  if (totalMass > 0) {
    const cx = sumX / totalMass;
    const cy = sumY / totalMass;

    const shiftX = Math.round(13.5 - cx);
    const shiftY = Math.round(13.5 - cy);

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const srcX = x - shiftX;
        const srcY = y - shiftY;
        if (srcX >= 0 && srcX < 28 && srcY >= 0 && srcY < 28) {
          final28[y * 28 + x] = grid28[srcY * 28 + srcX];
        }
      }
    }
  }

  // Draw onto the 28x28 preview canvas
  const previewImgData = previewCtx.createImageData(28, 28);
  for (let i = 0; i < 28 * 28; i++) {
    const val = final28[i];
    previewImgData.data[i * 4] = val;
    previewImgData.data[i * 4 + 1] = val;
    previewImgData.data[i * 4 + 2] = val;
    previewImgData.data[i * 4 + 3] = 255;
  }
  previewCtx.putImageData(previewImgData, 0, 0);

  // Normalize: (pixel / 255.0 - 0.5) / 0.5  => maps to [-1.0, 1.0]
  const inputTensorData = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    inputTensorData[i] = (final28[i] / 255.0 - 0.5) / 0.5;
  }

  return inputTensorData;
}

// Softmax calculation
function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExps);
}

// Run Inference
async function triggerPrediction() {
  if (!state.cnnSession && !state.rnnSession) return;

  const tensorData = preprocessCanvas(canvas);
  if (!tensorData) {
    resetOutput();
    return;
  }

  const tensor = new ort.Tensor("float32", tensorData, [1, 1, 28, 28]);
  const t0 = performance.now();

  if (state.activeModel === "cnn" && state.cnnSession) {
    const feeds = { input: tensor };
    const results = await state.cnnSession.run(feeds);
    const logits = Array.from(results.output.data);
    const probs = softmax(logits);
    const latency = (performance.now() - t0).toFixed(1);

    updateSingleDisplay("CNN", probs, latency);
  } else if (state.activeModel === "rnn" && state.rnnSession) {
    const feeds = { input: tensor };
    const results = await state.rnnSession.run(feeds);
    const logits = Array.from(results.output.data);
    const probs = softmax(logits);
    const latency = (performance.now() - t0).toFixed(1);

    updateSingleDisplay("RNN", probs, latency);
  } else if (state.activeModel === "duel") {
    let cnnProbs = null;
    let rnnProbs = null;

    if (state.cnnSession) {
      const res = await state.cnnSession.run({ input: tensor });
      cnnProbs = softmax(Array.from(res.output.data));
    }
    if (state.rnnSession) {
      const res = await state.rnnSession.run({ input: tensor });
      rnnProbs = softmax(Array.from(res.output.data));
    }

    const latency = (performance.now() - t0).toFixed(1);
    updateDuelDisplay(cnnProbs, rnnProbs, latency);
  }
}

// Update Single Model Display
function updateSingleDisplay(modelName, probs, latency) {
  latencyTag.textContent = `⚡ ${latency} ms (${modelName})`;

  let maxIdx = 0;
  let maxProb = probs[0];
  for (let i = 1; i < 10; i++) {
    if (probs[i] > maxProb) {
      maxProb = probs[i];
      maxIdx = i;
    }
  }

  topDigitEl.textContent = maxIdx;
  topConfidenceEl.textContent = `${(maxProb * 100).toFixed(1)}%`;
  activeModelLabel.textContent = `${modelName} Prediction`;

  // Update Bar Chart
  for (let i = 0; i <= 9; i++) {
    const fill = document.getElementById(`bar-fill-${i}`);
    const pct = document.getElementById(`bar-pct-${i}`);
    const row = document.getElementById(`bar-row-${i}`);
    const val = (probs[i] * 100).toFixed(1);

    fill.style.width = `${val}%`;
    pct.textContent = `${val}%`;

    if (i === maxIdx) {
      row.classList.add("active");
      fill.classList.add("top-pred");
    } else {
      row.classList.remove("active");
      fill.classList.remove("top-pred");
    }
  }
}

// Update Duel Display
function updateDuelDisplay(cnnProbs, rnnProbs, latency) {
  latencyTag.textContent = `⚔️ ${latency} ms (Duel)`;

  if (cnnProbs) {
    const maxCnn = cnnProbs.indexOf(Math.max(...cnnProbs));
    duelCnnDigit.textContent = maxCnn;
    duelCnnConf.textContent = `${(cnnProbs[maxCnn] * 100).toFixed(1)}%`;
  }
  if (rnnProbs) {
    const maxRnn = rnnProbs.indexOf(Math.max(...rnnProbs));
    duelRnnDigit.textContent = maxRnn;
    duelRnnConf.textContent = `${(rnnProbs[maxRnn] * 100).toFixed(1)}%`;
  }

  // Chart defaults to CNN probabilities in duel mode
  if (cnnProbs) {
    const maxIdx = cnnProbs.indexOf(Math.max(...cnnProbs));
    for (let i = 0; i <= 9; i++) {
      const fill = document.getElementById(`bar-fill-${i}`);
      const pct = document.getElementById(`bar-pct-${i}`);
      const row = document.getElementById(`bar-row-${i}`);
      const val = (cnnProbs[i] * 100).toFixed(1);

      fill.style.width = `${val}%`;
      pct.textContent = `${val}%`;

      if (i === maxIdx) {
        row.classList.add("active");
        fill.classList.add("top-pred");
      } else {
        row.classList.remove("active");
        fill.classList.remove("top-pred");
      }
    }
  }
}

// Reset Outputs
function resetOutput() {
  topDigitEl.textContent = "-";
  topConfidenceEl.textContent = "0.0%";
  duelCnnDigit.textContent = "-";
  duelCnnConf.textContent = "0.0%";
  duelRnnDigit.textContent = "-";
  duelRnnConf.textContent = "0.0%";
  latencyTag.textContent = "⚡ Ready";

  previewCtx.fillStyle = "#000000";
  previewCtx.fillRect(0, 0, 28, 28);

  for (let i = 0; i <= 9; i++) {
    const fill = document.getElementById(`bar-fill-${i}`);
    const pct = document.getElementById(`bar-pct-${i}`);
    const row = document.getElementById(`bar-row-${i}`);
    fill.style.width = "0%";
    pct.textContent = "0.0%";
    row.classList.remove("active");
    fill.classList.remove("top-pred");
  }
}

// Model Toggle Setup
function setupModelToggle() {
  modelToggleGroup.addEventListener("click", (e) => {
    const btn = e.target.closest(".toggle-btn");
    if (!btn) return;

    modelToggleGroup.querySelectorAll(".toggle-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    state.activeModel = btn.dataset.model;

    if (state.activeModel === "duel") {
      singleHero.classList.add("hidden");
      duelContainer.classList.remove("hidden");
    } else {
      singleHero.classList.remove("hidden");
      duelContainer.classList.add("hidden");
      activeModelLabel.textContent = `${state.activeModel.toUpperCase()} Prediction`;
    }

    triggerPrediction();
  });
}

// Tab Switching
function setupTabs() {
  drawTabBtn.addEventListener("click", () => {
    drawTabBtn.classList.add("active");
    uploadTabBtn.classList.remove("active");
    drawSection.classList.remove("hidden");
    uploadSection.classList.add("hidden");
    triggerPrediction();
  });

  uploadTabBtn.addEventListener("click", () => {
    uploadTabBtn.classList.add("active");
    drawTabBtn.classList.remove("active");
    uploadSection.classList.remove("hidden");
    drawSection.classList.add("hidden");
  });
}

// Image Uploader Setup
function setupImageUpload() {
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--accent-primary)";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "#4b5563";
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#4b5563";
    if (e.dataTransfer.files.length > 0) {
      handleImageFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleImageFile(e.target.files[0]);
    }
  });

  removeUploadBtn.addEventListener("click", () => {
    uploadPreviewWrapper.classList.add("hidden");
    dropzone.classList.remove("hidden");
    fileInput.value = "";
    resetOutput();
  });
}

function handleImageFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload a valid image file.");
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      uploadPreviewImg.src = img.src;
      dropzone.classList.add("hidden");
      uploadPreviewWrapper.classList.remove("hidden");

      // Draw onto canvas
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Fit image inside canvas
      const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
      const w = img.width * scale;
      const h = img.height * scale;
      const x = (canvas.width - w) / 2;
      const y = (canvas.height - h) / 2;
      ctx.drawImage(img, x, y, w, h);

      // Invert if image has white background
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      let avg = 0;
      for (let i = 0; i < imgData.data.length; i += 4) {
        avg += imgData.data[i];
      }
      avg = avg / (imgData.data.length / 4);

      if (avg > 127) {
        for (let i = 0; i < imgData.data.length; i += 4) {
          imgData.data[i] = 255 - imgData.data[i];
          imgData.data[i + 1] = 255 - imgData.data[i + 1];
          imgData.data[i + 2] = 255 - imgData.data[i + 2];
        }
        ctx.putImageData(imgData, 0, 0);
      }

      canvasHint.style.opacity = "0";
      triggerPrediction();
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// Bootstrap Application
document.addEventListener("DOMContentLoaded", () => {
  initBarChart();
  setupCanvas();
  setupModelToggle();
  setupTabs();
  setupImageUpload();
  initONNX();
});
