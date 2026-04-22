import init, { DocsegApp } from "./pkg/docseg_web.js";

const MODEL_URL = "./models/craft_mlt_25k.onnx";

const $ = (id) => document.getElementById(id);
const status = (msg) => { $("status").textContent = msg; };

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
  // Point onnxruntime-web at its wasm assets on the same CDN. Without this
  // it tries to fetch `ort-wasm-*.wasm` from our origin, which 404s.
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

  status("loading wasm…");
  await init();
  const app = new DocsegApp();

  status("fetching model (~83 MB, cached after first load)…");
  const modelRes = await fetch(MODEL_URL);
  if (!modelRes.ok) throw new Error(`model fetch: ${modelRes.status}`);
  const modelBytes = new Uint8Array(await modelRes.arrayBuffer());

  status("initializing WebGPU inference session…");
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "all",
  });

  status("ready. open an image.");

  $("file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    await runDetection(app, session, file);
  });

  $("export").addEventListener("click", () => {
    const bytes = app.exportZip();
    downloadBlob(new Blob([bytes], { type: "application/zip" }), "docseg.zip");
  });

  // Auto-load the sample image if present.
  try {
    const sample = await fetch("test_case1.png");
    if (sample.ok) {
      const blob = await sample.blob();
      await runDetection(app, session, blob);
    }
  } catch {
    /* no sample — user can pick one */
  }
}

async function runDetection(app, session, blob) {
  status("decoding + preprocessing…");
  const imgBytes = new Uint8Array(await blob.arrayBuffer());

  const t0 = performance.now();
  const pre = app.preprocessImage(imgBytes);
  const tPre = performance.now();

  status("running inference on WebGPU…");
  const input = new ort.Tensor(
    "float32",
    pre.tensor,
    [1, 3, pre.padded_h, pre.padded_w],
  );
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const outputs = await session.run({ [inputName]: input });
  const tInfer = performance.now();

  const out = outputs[outputName];
  // CRAFT output is [1, H/2, W/2, 2]; extract the region channel (idx 0).
  const { dims, data } = out;
  if (dims.length !== 4 || dims[3] !== 2) {
    throw new Error(`unexpected ort output shape ${JSON.stringify(dims)}`);
  }
  const hmH = dims[1];
  const hmW = dims[2];
  const region = new Float32Array(hmH * hmW);
  for (let i = 0; i < region.length; i++) {
    region[i] = data[i * 2];
  }

  status("postprocessing…");
  const detection = app.postprocess(
    region,
    hmW,
    hmH,
    pre.scale,
    pre.padded_w,
    pre.padded_h,
    pre.original_w,
    pre.original_h,
  );
  const tPost = performance.now();

  status("rendering…");
  const htmlImg = await blobToHtmlImage(blob);
  const canvas = $("canvas");
  canvas.width = detection.image.width;
  canvas.height = detection.image.height;
  const ctx = canvas.getContext("2d");
  app.paint(ctx, htmlImg);

  canvas.onclick = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left) * (canvas.width / rect.width);
    const y = (ev.clientY - rect.top) * (canvas.height / rect.height);
    const id = app.hit(x, y);
    if (id < 0) return;
    const png = app.cropPng(id);
    downloadBlob(new Blob([png], { type: "image/png" }), `glyph_${id}.png`);
  };

  $("export").disabled = false;
  status(
    `detected ${detection.boxes.length} characters | ` +
    `preprocess ${(tPre - t0).toFixed(0)}ms, ` +
    `inference ${(tInfer - tPre).toFixed(0)}ms, ` +
    `postprocess ${(tPost - tInfer).toFixed(0)}ms`,
  );
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
