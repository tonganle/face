const resultBox = document.querySelector("#resultBox");
const peopleList = document.querySelector("#peopleList");
const personCount = document.querySelector("#personCount");
const embeddingCount = document.querySelector("#embeddingCount");
const modelState = document.querySelector("#modelState");

function showResult(payload) {
  resultBox.textContent = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(payload.detail || response.statusText);
  }
  return payload;
}

async function refresh() {
  const [health, people] = await Promise.all([
    requestJson("/api/health"),
    requestJson("/api/people"),
  ]);
  personCount.textContent = health.persons;
  embeddingCount.textContent = health.embeddings;
  modelState.textContent = health.model_loaded ? "Loaded" : "Idle";
  peopleList.innerHTML = "";
  if (!people.length) {
    peopleList.textContent = "No identities yet";
    return;
  }
  for (const person of people) {
    const row = document.createElement("div");
    row.className = "person";
    const name = person.display_name || person.name;
    const key = person.identity_key || person.id;
    row.innerHTML = `<span>${name}</span><small>ID: ${key} · ${person.embedding_count} embeddings</small>`;
    peopleList.appendChild(row);
  }
}

document.querySelector("#refreshBtn").addEventListener("click", async () => {
  try {
    await refresh();
    showResult("Status refreshed");
  } catch (error) {
    showResult(`Refresh failed: ${error.message}`);
  }
});

document.querySelector("#pathTrainForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  showResult("Training dataset, please wait...");
  try {
    const payload = await requestJson("/api/train/path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_path: document.querySelector("#datasetPath").value }),
    });
    showResult(payload);
    await refresh();
  } catch (error) {
    showResult(`Training failed: ${error.message}`);
  }
});

document.querySelector("#zipTrainForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.querySelector("#zipFile").files[0];
  if (!file) {
    showResult("Please choose a zip file");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  showResult("Uploading and training, please wait...");
  try {
    const payload = await requestJson("/api/train/upload", { method: "POST", body: formData });
    showResult(payload);
    await refresh();
  } catch (error) {
    showResult(`Training failed: ${error.message}`);
  }
});

document.querySelector("#recognizeForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.querySelector("#recognizeFile").files[0];
  if (!file) {
    showResult("Please choose an image");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  showResult("Recognizing...");
  try {
    const payload = await requestJson("/api/recognize", { method: "POST", body: formData });
    showResult(payload);
  } catch (error) {
    showResult(`Recognition failed: ${error.message}`);
  }
});

refresh().catch((error) => showResult(`Initialization failed: ${error.message}`));
