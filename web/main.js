// docseg browser demo.
// The Rust side (wasm) owns preprocess, postprocess, reading order,
// overlay painting, hit-test, PNG crop, and zip export. JS orchestrates:
// fetch model → ort.InferenceSession on WebGPU → preprocess → inference →
// extract region+affinity channels → postprocess → paint → wire UI.

const BUILD_TAG = `${Date.now()}`;
const pkgMod = await import(`./pkg/docseg_web.js?t=${BUILD_TAG}`);
const { DocsegApp } = pkgMod;
const init = pkgMod.default;
const toolsMod = await import(`./tools.js?t=${BUILD_TAG}`);

// The CRAFT ONNX lives at this URL. Swap to a HuggingFace resolve URL
// (huggingface.co/<user>/<repo>/resolve/main/craft_mlt_25k.onnx) once the
// model is hosted there — the demo and the GH Pages build will fetch from
// the same place.
const MODEL_URL =
  (window.DOCSEG_MODEL_URL ?? "./models/craft_mlt_25k.onnx");

const $ = (id) => document.getElementById(id);
const status = (msg) => { $("status").textContent = msg; };

// Mutable session state, held in the module closure.
const state = {
  app: null,
  session: null,
  lastImageBlob: null,
  lastImage: null,
  lastDetection: null,
  mode: "select",
  selectedId: -1,
  drawSequence: [],
  highlightId: -1,
};

async function main() {
  if (!("gpu" in navigator)) {
    const banner = $("banner");
    banner.hidden = false;
    banner.textContent =
      "WebGPU is required. This demo needs Chrome 113+, Safari 17.4+, or Firefox Nightly with WebGPU enabled.";
    status("");
    return;
  }
  if (typeof ort === "undefined") {
    throw new Error("onnxruntime-web failed to load from CDN");
  }
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

  status("loading wasm…");
  await init(`./pkg/docseg_web_bg.wasm?t=${BUILD_TAG}`);
  state.app = new DocsegApp();

  status("fetching model (~83 MB; cached after first load)…");
  const modelRes = await fetch(MODEL_URL);
  if (!modelRes.ok) throw new Error(`model fetch: ${modelRes.status}`);
  const modelBytes = new Uint8Array(await modelRes.arrayBuffer());

  status("initializing WebGPU inference session…");
  state.session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "all",
  });

  wireUi();
  status("ready. open an image.");

  try {
    const sample = await fetch("test_case1.png");
    if (sample.ok) {
      const blob = await sample.blob();
      await runDetection(blob);
    }
  } catch { /* no sample */ }
}

function wireUi() {
  state.selectedId = -1;
  state.drawSequence = [];
  toolsMod.initTools({
    canvas: $("canvas"),
    app: state.app,
    state,
    onChange: () => {
      renderRibbon();
      repaint();
    },
  });
  const resetOrder = $("tool-reset-order");
  if (resetOrder) {
    resetOrder.addEventListener("click", () => {
      state.drawSequence = [];
      const order = state.app.computeReadingOrder($("direction").value);
      if (state.lastDetection) {
        state.lastDetection.order = Array.from(order);
      }
      renderRibbon();
      repaint();
    });
  }

  $("file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    await runDetection(f);
  });
  $("export").addEventListener("click", () => {
    const bytes = state.app.exportZip();
    downloadBlob(new Blob([bytes], { type: "application/zip" }), "docseg.zip");
  });
  $("show-order").addEventListener("change", repaint);
  $("direction").addEventListener("change", applyReadingOrder);

  // Detection sliders: live-update postprocess (cheap, no re-inference).
  for (const id of ["region-threshold", "affinity-threshold", "erosion-px", "min-area"]) {
    const el = $(id);
    const out = $(`${id}-out`);
    el.addEventListener("input", () => {
      out.textContent = el.type === "range" && Number(el.step) < 1
        ? Number(el.value).toFixed(2)
        : el.value;
      recomputeFromCachedHeatmap();
    });
  }
  $("axis-aligned").addEventListener("change", recomputeFromCachedHeatmap);
}

let lastHeatmap = null; // { region: Float32Array, affinity: Float32Array, hmW, hmH, pre }

async function runDetection(blob) {
  state.lastImageBlob = blob;
  state.drawSequence = [];
  status("decoding + preprocessing…");
  const imgBytes = new Uint8Array(await blob.arrayBuffer());

  const t0 = performance.now();
  const pre = state.app.preprocessImage(imgBytes);
  const tPre = performance.now();

  status("running inference on WebGPU…");
  const input = new ort.Tensor(
    "float32",
    pre.tensor,
    [1, 3, pre.padded_h, pre.padded_w],
  );
  const inputName = state.session.inputNames[0];
  const outputName = state.session.outputNames[0];
  const outputs = await state.session.run({ [inputName]: input });
  const tInfer = performance.now();

  const out = outputs[outputName];
  const { dims, data } = out;
  if (dims.length !== 4 || dims[3] !== 2) {
    throw new Error(`unexpected ort output shape ${JSON.stringify(dims)}`);
  }
  const hmH = dims[1];
  const hmW = dims[2];
  const plane = hmH * hmW;
  const region = new Float32Array(plane);
  const affinity = new Float32Array(plane);
  for (let i = 0; i < plane; i++) {
    region[i] = data[i * 2];
    affinity[i] = data[i * 2 + 1];
  }
  lastHeatmap = { region, affinity, hmW, hmH, pre };

  const htmlImg = await blobToHtmlImage(blob);
  state.lastImage = htmlImg;

  runPostprocess();
  status(
    `detected ${state.lastDetection.boxes.length} characters | ` +
    `preprocess ${(tPre - t0).toFixed(0)}ms, ` +
    `inference ${(tInfer - tPre).toFixed(0)}ms`,
  );
  $("export").disabled = false;
}

function recomputeFromCachedHeatmap() {
  if (!lastHeatmap || !state.lastImage) return;
  runPostprocess();
}

function runPostprocess() {
  const opts = readOptsFromUi();
  const t0 = performance.now();
  const { region, affinity, hmW, hmH, pre } = lastHeatmap;
  state.lastDetection = state.app.postprocess(
    region, affinity, hmW, hmH,
    pre.scale, pre.padded_w, pre.padded_h, pre.original_w, pre.original_h,
    opts.regionThreshold,
    opts.affinityThreshold,
    opts.erosionPx,
    opts.axisAligned,
    opts.minArea,
    $("direction").value,
  );
  const t1 = performance.now();
  state.drawSequence = [];
  // Resize canvas to image bounds, then repaint and refresh ribbon.
  const canvas = $("canvas");
  canvas.width = state.lastDetection.image.width;
  canvas.height = state.lastDetection.image.height;
  renderRibbon();
  repaint();
  status(
    `detected ${state.lastDetection.boxes.length} characters | postprocess ${(t1 - t0).toFixed(0)}ms`,
  );
}

function applyReadingOrder() {
  if (!state.lastDetection) return;
  if (state.mode === "order" && state.drawSequence.length > 0) {
    state.app.setCustomOrder(new Uint32Array(state.drawSequence));
    state.lastDetection.order = [...state.drawSequence];
  } else {
    const order = state.app.computeReadingOrder($("direction").value);
    state.lastDetection.order = Array.from(order);
  }
  renderRibbon();
  repaint();
}

function readOptsFromUi() {
  return {
    regionThreshold: Number($("region-threshold").value),
    affinityThreshold: Number($("affinity-threshold").value),
    erosionPx: Number($("erosion-px").value),
    minArea: Number($("min-area").value),
    axisAligned: $("axis-aligned").checked,
  };
}

function repaint() {
  if (!state.lastImage) return;
  const canvas = $("canvas");
  const ctx = canvas.getContext("2d");
  state.app.paint(
    ctx,
    state.lastImage,
    $("show-order").checked,
    state.highlightId,
  );
}

function renderRibbon() {
  const ribbon = $("ribbon");
  ribbon.innerHTML = "";
  if (!state.lastDetection) return;
  const order = state.lastDetection.order ?? [];
  const frag = document.createDocumentFragment();
  for (let rank = 0; rank < order.length; rank++) {
    const id = order[rank];
    const png = state.app.cropPng(id);
    const url = URL.createObjectURL(new Blob([png], { type: "image/png" }));
    const cell = document.createElement("div");
    cell.className = "ribbon-cell";
    cell.dataset.id = String(id);
    cell.innerHTML =
      `<img src="${url}" alt="glyph ${id}" />` +
      `<span class="rank">${rank + 1}</span>`;
    cell.addEventListener("mouseenter", () => {
      state.highlightId = id;
      cell.classList.add("hover");
      repaint();
    });
    cell.addEventListener("mouseleave", () => {
      state.highlightId = -1;
      cell.classList.remove("hover");
      repaint();
    });
    frag.appendChild(cell);
  }
  ribbon.appendChild(frag);
}

function blobToHtmlImage(blob) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(e);
    img.src = url;
  });
}

function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

main().catch((e) => {
  console.error(e);
  status(`error: ${e.message}`);
});
