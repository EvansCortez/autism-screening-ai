"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("screening-form");
const routes = new Set(["welcome", "intake", "summary", "example", "research", "about"]);
const defaults = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0];
let lastResult = null;
let controller = null;
let busy = false;
let evidenceLoaded = false;
let evidenceLoading = false;
const mobileQuery = window.matchMedia("(max-width: 600px)");
if (mobileQuery.matches) for (const id of ["dataset-fields", "notes-fields"]) byId(id).open = false;
mobileQuery.addEventListener("change", (event) => {
  // Expand on desktop; never collapse a user's focused controls during a resize.
  if (!event.matches) for (const id of ["dataset-fields", "notes-fields"]) byId(id).open = true;
});

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

for (let index = 1; index <= 10; index += 1) {
  const field = element("div", undefined, "field");
  const label = element("label", `A${index} score`);
  const input = element("input");
  input.id = `A${index}_Score`;
  input.name = input.id;
  label.htmlFor = input.id;
  Object.assign(input, { type: "number", min: "0", max: "1", step: "1", required: true, value: String(defaults[index - 1]), inputMode: "numeric" });
  input.setAttribute("aria-describedby", "aq-help");
  field.append(label, input);
  byId("aq-fields").append(field);
}

function status(message) {
  byId("app-status").textContent = message;
}

function showRoute(focus = true) {
  const route = routes.has(location.hash.slice(1)) ? location.hash.slice(1) : "welcome";
  if (busy && route !== "intake") controller?.abort();
  document.querySelectorAll("[data-view]").forEach((section) => {
    section.hidden = section.dataset.view !== route;
  });
  document.querySelectorAll("nav a").forEach((link) => {
    if (link.hash === `#${route}`) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.title = `${{ welcome: "Welcome", intake: "Adult Data Demo", summary: "Live Summary", example: "Illustrative Example", research: "Research Evidence", about: "About the Project" }[route]} | ASD-Predict`;
  if (route === "summary") renderLiveSummary();
  if (route === "research") loadEvidence();
  if (focus) {
    document.querySelector(`[data-view="${route}"] h1`).focus();
    window.scrollTo(0, 0);
  }
}

function clearErrors() {
  byId("form-errors").replaceChildren();
  byId("form-errors").hidden = true;
  form.querySelectorAll("[aria-invalid]").forEach((field) => {
    field.removeAttribute("aria-invalid");
    field.removeAttribute("aria-errormessage");
    const descriptions = (field.getAttribute("aria-describedby") || "").split(" ").filter((id) => id && !id.startsWith("field-error-"));
    if (descriptions.length) field.setAttribute("aria-describedby", descriptions.join(" "));
    else field.removeAttribute("aria-describedby");
  });
}

function fieldLabel(id) {
  return document.querySelector(`label[for="${CSS.escape(id)}"]`)?.textContent || id;
}

function showErrors(errors) {
  const box = byId("form-errors");
  box.replaceChildren(element("strong", "The summary could not be generated."));
  const list = element("ul");
  for (const [index, { id, message }] of errors.entries()) {
    const item = element("li");
    const field = id ? byId(id) : null;
    if (field && form.contains(field)) {
      field.setAttribute("aria-invalid", "true");
      item.id = `field-error-${index}`;
      field.setAttribute("aria-errormessage", item.id);
      field.setAttribute("aria-describedby", `${field.getAttribute("aria-describedby") || ""} ${item.id}`.trim());
      for (const details of form.querySelectorAll("details")) if (details.contains(field)) details.open = true;
      const link = element("a", `${fieldLabel(id)}: ${message}`);
      link.href = `#${id}`;
      link.addEventListener("click", (event) => { event.preventDefault(); field.focus(); });
      item.append(link);
    } else item.textContent = message;
    list.append(item);
  }
  box.append(list);
  box.hidden = false;
  const first = errors.find(({ id }) => id && byId(id) && form.contains(byId(id)));
  if (first) byId(first.id).focus();
  else { box.tabIndex = -1; box.focus(); }
}

function validateForm() {
  clearErrors();
  const errors = [];
  for (const field of form.querySelectorAll("input, select, textarea")) {
    field.setCustomValidity(field.required && !field.value.trim() ? "Enter a nonblank value." : "");
    if (!field.checkValidity()) errors.push({ id: field.id, message: field.validationMessage });
  }
  if (errors.length) {
    showErrors(errors);
    status("Check the highlighted fields. No request was sent.");
    return false;
  }
  return true;
}

function basePayload() {
  const payload = {};
  for (let i = 1; i <= 10; i += 1) payload[`A${i}_Score`] = Number(byId(`A${i}_Score`).value);
  payload.age = Number(byId("age").value);
  for (const name of ["gender", "ethnicity", "jaundice", "country_of_res", "relation"]) payload[name] = byId(name).value.trim();
  return payload;
}

function reportPayload() {
  const payload = basePayload();
  for (const name of ["subject_name", "screener_name", "context", "caregiver_concerns", "transcript_excerpt", "prior_screening_status", "referral_status", "follow_up_plan"]) payload[name] = byId(name).value.trim();
  for (const name of ["observed_behaviors", "follow_up_responses"]) payload[name] = byId(name).value.split("\n").map((line) => line.trim()).filter(Boolean);
  return payload;
}

function card(title, text) {
  const article = element("article", undefined, "panel");
  article.append(element("h2", title));
  if (text !== undefined) article.append(element("p", text, "supplied-text"));
  return article;
}

function renderLiveSummary() {
  const target = byId("live-summary");
  target.replaceChildren();
  if (!lastResult) {
    target.append(card("No live result yet", "Submit the intake to view actual API output. The illustrative example is separate and is never used as a fallback result."));
    return;
  }
  const { data, metadata, modelOnly, stale } = lastResult;
  const result = modelOnly ? data : data.screening_result;
  const main = card(`${modelOnly ? metadata.subject_name : data.subject_name} / ${modelOnly ? metadata.context : data.context}`);
  main.prepend(element("p", `Live API result · ${modelOnly ? "/predict" : "/screening-report"} · educational prototype`, "eyebrow"));
  if (stale) main.prepend(element("p", "Inputs have changed since this result. Submit again to update it; these values still belong to the previous submission.", "stale-notice"));
  const metrics = element("dl", undefined, "metrics");
  for (const [label, value] of [["Item total", `${result.sum_score} / 10`], ["Model dataset-label output", `${result.dataset_label} label`], ["Clinical probability", "Not established"]]) {
    const group = element("div");
    group.append(element("dt", label), element("dd", value));
    metrics.append(group);
  }
  main.append(metrics, element("h2", "Reading this output"), element("p", `Observed >=7 dataset rule: ${result.observed_rule_label}. This rule and the saved model are separate; they can disagree. The documented referral-consideration threshold is >=6. Neither is a diagnosis.`));
  const details = element("details");
  details.append(element("summary", "Model label score and interpretation"), element("p", `Class-YES model score: ${result.model_label_score.toFixed(4)} (0 to 1). This is an uncalibrated model output, not a clinical probability of autism. Oversampling and training-label construction affect its interpretation.`));
  main.append(details);
  target.append(main);
  if (modelOnly) target.append(card("Supporting notes", "Model-only request: no notes or follow-up text were sent to the prediction endpoint."));
  else {
    const report = data.report;
    const notes = card("Supporting notes", "Supplied text only, not diagnostic findings. These fields do not enter the model.");
    notes.append(element("h3", "Screening notes"), element("p", report.caregiver_concerns, "supplied-text"), element("h3", "Transcript excerpt"), element("p", report.transcript_excerpt, "supplied-text"));
    if (metadata.observed_behaviors.length) notes.append(element("h3", "Observation tags (supplied)"), element("p", metadata.observed_behaviors.join("\n"), "supplied-text"));
    target.append(notes);
    const followUp = report.follow_up_summary;
    const workflow = card("Follow-up workflow", "Supplied workflow context, not a recommendation generated from the model score.");
    for (const [label, value] of [["Prior screening", followUp.prior_screening_status], ["Referral", followUp.referral_status], ["Plan", followUp.follow_up_plan]]) workflow.append(element("p", `${label}: ${value}`, "supplied-text"));
    if (metadata.follow_up_responses.length) workflow.append(element("p", followUp.follow_up_items.join("\n"), "supplied-text"));
    target.append(workflow);
  }
  target.append(card("Model limitations", "Independent diagnostic outcomes and clinical external validation are not established. Demographic associations cannot establish their cause. A match with these screening labels is not clinically validated diagnostic accuracy."));
}

function normalizeResult(data, modelOnly) {
  const result = modelOnly ? data : data.screening_result;
  if (!result || typeof result.asd_traits_detected !== "boolean" || !Number.isInteger(result.sum_score) || result.sum_score < 0 || result.sum_score > 10) throw new Error("invalid-response");
  const score = result.model_label_score ?? result.probability;
  if (!Number.isFinite(score) || score < 0 || score > 1) throw new Error("invalid-response");
  const label = result.asd_traits_detected ? "YES" : "NO";
  const ruleLabel = result.sum_score >= 7 ? "YES" : "NO";
  if (result.dataset_label !== undefined && result.dataset_label !== label) throw new Error("invalid-response");
  if (result.observed_rule_label !== undefined && result.observed_rule_label !== ruleLabel) throw new Error("invalid-response");
  // Legacy response aliases remain usable; never substitute the Figma illustration.
  result.dataset_label = result.dataset_label ?? label;
  result.observed_rule_label = result.observed_rule_label ?? ruleLabel;
  result.model_label_score = score;
  if (!modelOnly && (!data.report?.follow_up_summary || !Array.isArray(data.report.follow_up_summary.follow_up_items))) throw new Error("invalid-response");
  return data;
}

async function submit(modelOnly) {
  if (busy || !validateForm()) return;
  const payload = modelOnly ? basePayload() : reportPayload();
  const metadata = reportPayload();
  lastResult = null;
  const path = modelOnly ? "/predict" : "/screening-report";
  busy = true;
  controller = new AbortController();
  const current = controller;
  let focusError = null;
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; current.abort(); }, 20000);
  form.setAttribute("aria-busy", "true");
  form.querySelectorAll("fieldset, button").forEach((control) => { control.disabled = true; });
  byId("submit-report").textContent = modelOnly ? "Generate demo summary" : "Generating summary...";
  byId("predict-only").textContent = modelOnly ? "Running model..." : "Run model only";
  byId("intake-status").textContent = "Waiting for the API...";
  status("Request in progress. Inputs are temporarily locked.");
  try {
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: current.signal });
    let data;
    try { data = await response.json(); } catch { throw new Error("invalid-response"); }
    if (current.signal.aborted) return;
    if (!response.ok) {
      const errors = Array.isArray(data.detail) ? data.detail.map((error) => ({ id: error.loc?.at(-1), message: String(error.msg || "Invalid input") })) : [{ message: response.status === 503 ? "The saved model is unavailable. No result was generated. Try again when the service is restored." : `API request failed (${response.status}). ${typeof data.detail === "string" ? data.detail.slice(0, 500) : "Please review inputs or retry."}` }];
      showErrors(errors);
      focusError = errors.find(({ id }) => id && byId(id))?.id;
      status("API request failed. No live result is available; inputs are preserved.");
      byId("intake-status").textContent = "Request failed. You can retry.";
      return;
    }
    lastResult = { data: normalizeResult(data, modelOnly), metadata, modelOnly, stale: false };
    busy = false;
    byId("intake-status").textContent = "Live result ready";
    status("Live API summary ready. This is not the illustrative example.");
    location.hash = "summary";
  } catch (error) {
    if (current.signal.aborted && !timedOut) {
      status("Request cancelled on navigation. Inputs are preserved; no result was generated.");
      byId("intake-status").textContent = "Request cancelled";
    } else {
      const message = timedOut ? "The API did not respond within 20 seconds. No result was generated. Please retry." : error.message === "invalid-response" ? "The API returned an unexpected response. No result was generated." : "The API could not be reached. No result was generated. Check the connection and retry.";
      showErrors([{ message }]);
      status(message);
      byId("intake-status").textContent = "Request failed. You can retry.";
    }
  } finally {
    clearTimeout(timeout);
    busy = false;
    controller = null;
    form.removeAttribute("aria-busy");
    form.querySelectorAll("fieldset, button").forEach((control) => { control.disabled = false; });
    byId("submit-report").textContent = "Generate demo summary";
    byId("predict-only").textContent = "Run model only";
    if (focusError) byId(focusError).focus();
  }
}

async function loadEvidence(force = false) {
  if ((evidenceLoaded && !force) || evidenceLoading) return;
  evidenceLoading = true;
  byId("retry-evidence").hidden = true;
  byId("evidence-status").textContent = "Loading hash-verified committed evidence...";
  byId("research-results").setAttribute("aria-busy", "true");
  try {
    const response = await fetch("/research-evidence", { signal: AbortSignal.timeout(15000) });
    if (!response.ok) throw new Error("Evidence unavailable");
    const data = await response.json();
    if (!/^[a-f0-9]{40}$/.test(data.version) || !Array.isArray(data.metrics)) throw new Error("Invalid evidence version");
    const target = byId("research-results");
    target.replaceChildren();
    const wrap = element("div", undefined, "table-wrap");
    const table = element("table");
    table.append(element("caption", `Adult dataset: ${data.rows} records. ${data.folds} grouped folds; means over seeds ${data.seeds.join(", ")}.`));
    const head = element("thead");
    const heading = element("tr");
    for (const label of ["Method", "Mean label agreement"]) { const cell = element("th", label); cell.scope = "col"; heading.append(cell); }
    head.append(heading);
    const body = element("tbody");
    for (const metric of data.metrics) {
      const row = element("tr");
      row.append(element("td", metric.name), element("td", `${(metric.accuracy * 100).toFixed(2)}%`));
      body.append(row);
    }
    table.append(head, body);
    wrap.append(table);
    target.append(wrap, element("p", data.interpretation, "small muted"));
    const residual = data.metrics.find((metric) => metric.method === "logistic/non_questionnaire");
    target.append(element("p", `Non-questionnaire logistic ROC-AUC: ${residual.roc_auc.toFixed(4)} in grouped five-fold validation; ${data.country_held_out_auc.toFixed(4)} when each recorded country is held out. These are different validation tasks, not causal findings.`));
    const source = element("a", `Inspect source commit ${data.version.slice(0, 7)}`);
    source.href = `https://github.com/EvansCortez/autism-screening-ai/tree/${data.version}/research`;
    target.append(source);
    const figures = byId("research-figures");
    figures.replaceChildren();
    for (const item of data.figures) {
      if (!/^fig[1-6]_[a-z_]+$/.test(item.id)) throw new Error("Invalid figure reference");
      const figure = element("figure");
      const image = element("img");
      image.src = `/research-figures/${item.id}.png`;
      image.alt = item.description;
      image.loading = "lazy";
      image.addEventListener("error", () => { image.hidden = true; figure.append(element("p", "Figure could not be loaded. Inspect the pinned source instead.", "small")); });
      figure.append(element("h3", item.title), image, element("figcaption", item.description));
      figures.append(figure);
    }
    byId("evidence-status").textContent = `Source: committed experiment outputs · research/label-circularity · ${data.version.slice(0, 7)}. File hashes verified by the server.`;
    evidenceLoaded = true;
  } catch {
    byId("research-results").replaceChildren();
    byId("research-figures").replaceChildren();
    byId("evidence-status").textContent = "Research evidence is unavailable or does not match the pinned version. No illustrative metrics are substituted. The live demo is separate.";
    byId("retry-evidence").hidden = false;
    evidenceLoaded = false;
  } finally {
    evidenceLoading = false;
    byId("research-results").removeAttribute("aria-busy");
  }
}

form.addEventListener("submit", (event) => { event.preventDefault(); submit(false); });
byId("predict-only").addEventListener("click", () => submit(true));
byId("retry-evidence").addEventListener("click", () => loadEvidence(true));
form.addEventListener("input", () => { clearErrors(); if (lastResult) lastResult.stale = true; });
window.addEventListener("hashchange", () => showRoute());
document.querySelector(".skip-link").addEventListener("click", (event) => { event.preventDefault(); byId("main").focus(); });
showRoute(false);
