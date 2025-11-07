// script.js - usa ONNX Runtime Web para hacer inferencia
let session = null;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultadoEl = document.getElementById('resultado');
const labelsEl = document.getElementById('labels');

// etiquetas Fashion-MNIST
const LABELS = [
  "0: T-shirt/top", "1: Trouser", "2: Pullover", "3: Dress", "4: Coat",
  "5: Sandal", "6: Shirt", "7: Sneaker", "8: Bag", "9: Ankle boot"
];

// fondo negro
ctx.fillStyle = 'black';
ctx.fillRect(0,0,canvas.width, canvas.height);

// estilo de dibujo
ctx.lineWidth = 24;
ctx.lineCap = "round";
ctx.strokeStyle = "white";
let drawing = false;
canvas.addEventListener('mousedown', (e)=>{ drawing=true; ctx.beginPath(); ctx.moveTo(e.offsetX, e.offsetY); });
canvas.addEventListener('mousemove', (e)=>{ if(!drawing) return; ctx.lineTo(e.offsetX, e.offsetY); ctx.stroke(); });
canvas.addEventListener('mouseup', ()=>drawing=false);
canvas.addEventListener('mouseleave', ()=>drawing=false);

// botones
document.getElementById('limpiar').onclick = () => {
  ctx.fillStyle = 'black';
  ctx.fillRect(0,0,canvas.width, canvas.height);
  resultadoEl.innerText = '';
  labelsEl.innerHTML = '';
};

// Cargar modelo ONNX (de folder model/fashion_mnist.onnx)
async function cargarModelo() {
  try {
    resultadoEl.innerText = "Cargando modelo ONNX...";
    // create session desde ruta relativa
    session = await ort.InferenceSession.create('model/fashion_mnist.onnx', { executionProviders: ['wasm','webgl'] });
    resultadoEl.innerText = "✅ Modelo ONNX cargado";
  } catch (e) {
    console.error("Error cargando ONNX:", e);
    resultadoEl.innerText = "❌ Error cargando modelo: " + e.message;
  }
}

// Preprocesar canvas -> Float32Array [1,1,28,28] (NCHW)
function preprocessCanvas() {
  // crear un canvas temporario 28x28
  const temp = document.createElement('canvas');
  temp.width = 28; temp.height = 28;
  const tctx = temp.getContext('2d');

  // copiar y escalar desde el canvas principal
  tctx.drawImage(canvas, 0, 0, 28, 28);

  // obtener px
  const imgData = tctx.getImageData(0,0,28,28).data;
  const data = new Float32Array(1*1*28*28); // NCHW
  let ptr = 0;
  for (let y=0; y<28; y++){
    for (let x=0; x<28; x++){
      const i = (y*28 + x)*4;
      // imgData = rgba, 0 black -> 0, white -> 255
      // queremos normalized 0..1 where white=1, black=0
      // promedio de r,g,b (canvas white on black)
      const r = imgData[i], g = imgData[i+1], b = imgData[i+2];
      const gray = (r + g + b) / 3;
      // invert: on our canvas white strokes (255) => model expects white=1
      const norm = gray / 255.0;
      // For NCHW, index is [0,0,y,x]
      data[ptr++] = norm;
    }
  }
  return data;
}

// Ejecutar inferencia
document.getElementById('predecir').onclick = async () => {
  if (!session) { alert("Modelo no cargado aún"); return; }
  resultadoEl.innerText = "Predicting...";
  try {
    const inputData = preprocessCanvas(); // Float32Array length=784
    // ONNX Runtime expects TypedArray and shape
    const tensor = new ort.Tensor('float32', inputData, [1,1,28,28]);
    const feeds = { input: tensor }; // 'input' es el nombre que usamos al exportar
    const results = await session.run(feeds);
    // 'output' es el nombre dado al exportar
    const outputData = results.output.data; // Float32Array of length 10
    // convertir a array y obtener top1
    const arr = Array.from(outputData);
    const maxIdx = arr.indexOf(Math.max(...arr));
    resultadoEl.innerText = `Predicción: ${maxIdx} — ${LABELS[maxIdx]}`;
    // mostrar probabilidades
    labelsEl.innerHTML = arr.map((p,i)=> `${i}: ${LABELS[i].split(': ')[1]} — ${ (p*100).toFixed(1) }%`).join('<br>');
  } catch (e) {
    console.error("Error en predicción:", e);
    resultadoEl.innerText = "Error en predicción: " + e.message;
  }
};

// cargar modelo al inicio
cargarModelo();
