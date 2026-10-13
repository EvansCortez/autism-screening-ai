const { test, before, after } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { chromium } = require("playwright");

const base = process.env.UI_BASE_URL || "http://127.0.0.1:8787";
const screenshotDir = path.resolve(__dirname, "../docs/ui/screenshots");
let browser;

before(async () => {
  let ready = false;
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline && !ready) {
    try {
      const response = await fetch(`${base}/health`, { signal: AbortSignal.timeout(2000) });
      ready = response.ok && (await response.json()).model_loaded === true;
    } catch {}
    if (!ready) await new Promise((resolve) => setTimeout(resolve, 500));
  }
  assert.ok(ready, `Server/model not ready at ${base}. Inspect server logs and /health before running UI checks.`);
  browser = await chromium.launch({ headless: true, ...(process.env.UI_BROWSER_PATH ? { executablePath: process.env.UI_BROWSER_PATH } : {}) });
  await fs.mkdir(screenshotDir, { recursive: true });
});
after(async () => { await browser?.close(); });

async function pageFor(width = 1200) {
  const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: "reduce" });
  const page = await context.newPage();
  await page.goto(base);
  await page.evaluate(() => document.fonts.ready);
  return { page, context };
}

async function overflow(page) {
  return page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
}

test("welcome, navigation, real report and evidence match the data flow", async () => {
  const { page, context } = await pageFor();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.screenshot({ path: path.join(screenshotDir, "welcome-desktop.png"), fullPage: true });
  await page.getByRole("link", { name: "Open the demo" }).click();
  await page.waitForURL("**/#intake");
  await page.screenshot({ path: path.join(screenshotDir, "intake-desktop.png"), fullPage: true });
  let request;
  page.on("request", (event) => { if (event.url().endsWith("/screening-report")) request = event.postDataJSON(); });
  await page.getByLabel("Screening notes", { exact: true }).fill("Fictional supporting note for the interface demonstration.");
  const responsePromise = page.waitForResponse((response) => response.url().endsWith("/screening-report"));
  await page.getByRole("button", { name: "Generate demo summary", exact: true }).click();
  const response = await responsePromise;
  assert.equal(response.status(), 200);
  const data = await response.json();
  await page.waitForURL("**/#summary");
  const result = data.screening_result;
  assert.match(await page.locator("#live-summary").innerText(), new RegExp(`${result.sum_score} / 10`));
  assert.match(await page.locator("#live-summary").innerText(), new RegExp(`${result.dataset_label} label`));
  assert.equal(request.A1_Score, 1);
  assert.equal(request.age, 27);
  assert.equal(request.caregiver_concerns, "Fictional supporting note for the interface demonstration.");
  for (const field of ["screener_name", "context", "gender", "ethnicity", "jaundice", "country_of_res", "relation", "prior_screening_status", "referral_status", "follow_up_plan", "observed_behaviors", "follow_up_responses"]) assert.ok(Object.hasOwn(request, field));
  assert.equal(await page.locator('[data-view="summary"]').getByText("Illustrative example", { exact: true }).count(), 0);
  await page.screenshot({ path: path.join(screenshotDir, "summary-desktop.png"), fullPage: true });
  await page.getByRole("link", { name: "Research", exact: true }).click();
  await page.waitForFunction(() => document.getElementById("evidence-status").textContent.includes("File hashes verified"));
  assert.match(await page.locator("#research-results").innerText(), /73.15%/);
  assert.match(await page.locator("#research-results").innerText(), /0.6865/);
  assert.equal(await page.locator("#research-figures img").count(), 6);
  await page.locator("#research-figures").scrollIntoViewIfNeeded();
  // Explicitly load all lazy figures before capturing the committed evidence screen.
  await page.locator("#research-figures img").evaluateAll((images) => images.forEach((image) => { image.loading = "eager"; }));
  await page.waitForFunction(() => [...document.querySelectorAll("#research-figures img")].every((image) => image.complete && image.naturalWidth > 0));
  await page.screenshot({ path: path.join(screenshotDir, "research-desktop.png"), fullPage: true });
  assert.equal(await overflow(page), false);
  assert.deepEqual(errors, []);
  await context.close();
});

test("illustrative summary is separate; zero scores produce actual model output", async () => {
  const { page, context } = await pageFor();
  await page.goto(`${base}/#example`);
  assert.match(await page.locator('[data-view="example"]').innerText(), /not a live model result/);
  await page.getByRole("link", { name: "Return to intake" }).click();
  for (let i = 1; i <= 10; i++) await page.locator(`#A${i}_Score`).fill("0");
  const responsePromise = page.waitForResponse((response) => response.url().endsWith("/predict"));
  await page.getByRole("button", { name: "Run model only", exact: true }).click();
  const result = await (await responsePromise).json();
  await page.waitForURL("**/#summary");
  assert.equal(result.sum_score, 0);
  assert.match(await page.locator("#live-summary").innerText(), /0 \/ 10/);
  assert.match(await page.locator("#live-summary").innerText(), new RegExp(`${result.dataset_label} label`));
  await page.getByRole("link", { name: "Return to intake" }).click();
  await page.getByLabel("Participant name", { exact: true }).fill("Edited fictional name");
  await page.goto(`${base}/#summary`);
  assert.match(await page.locator("#live-summary").innerText(), /Inputs have changed/);
  await context.close();
});

test("both actions validate inputs and expose keyboard-accessible errors", async () => {
  const { page, context } = await pageFor();
  await page.goto(`${base}/#intake`);
  let requests = 0;
  page.on("request", (request) => { if (["/predict", "/screening-report"].some((route) => request.url().endsWith(route))) requests++; });
  await page.getByLabel("Age (18+)", { exact: true }).fill("17");
  await page.getByRole("button", { name: "Run model only", exact: true }).click();
  assert.equal(requests, 0);
  assert.equal(await page.locator("#age").getAttribute("aria-invalid"), "true");
  assert.equal(await page.evaluate(() => document.activeElement.id), "age");
  assert.match(await page.locator("#age").getAttribute("aria-describedby"), /field-error-/);
  await page.getByLabel("Age (18+)", { exact: true }).fill("27");
  await page.locator("#A1_Score").fill("2");
  await page.getByRole("button", { name: "Generate demo summary", exact: true }).click();
  assert.equal(requests, 0);
  await page.locator("#A1_Score").fill("1");
  await page.getByLabel("Participant name", { exact: true }).fill("   ");
  await page.getByRole("button", { name: "Generate demo summary", exact: true }).click();
  assert.equal(requests, 0);
  assert.equal(await page.locator("#subject_name").getAttribute("aria-invalid"), "true");
  assert.equal(await page.evaluate(() => document.activeElement.id), "subject_name");
  await page.goto(base);
  await page.keyboard.press("Tab");
  assert.equal(await page.getByRole("link", { name: "Skip to content" }).evaluate((link) => link === document.activeElement), true);
  await page.keyboard.press("Enter");
  assert.equal(await page.evaluate(() => document.activeElement.id), "main");
  await context.close();
});

test("untrusted participant and notes text stays text, not HTML", async () => {
  const { page, context } = await pageFor();
  await page.goto(`${base}/#intake`);
  const attack = '<img src=x onerror="window.__injected=1">';
  await page.getByLabel("Participant name", { exact: true }).fill(attack);
  await page.getByLabel("Screening notes", { exact: true }).fill(attack);
  await page.getByRole("button", { name: "Generate demo summary", exact: true }).click();
  await page.waitForURL("**/#summary");
  assert.match(await page.locator("#live-summary").innerText(), /<img src=x/);
  assert.equal(await page.locator("#live-summary img").count(), 0);
  assert.equal(await page.evaluate(() => window.__injected), undefined);
  await context.close();
});

test("loading locks duplicate submissions; 503 and network errors never substitute an example", async () => {
  const { page, context } = await pageFor();
  await page.goto(`${base}/#intake`);
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  await page.route("**/screening-report", async (route) => { await gate; await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Model unavailable" }) }); });
  await page.getByRole("button", { name: "Generate demo summary", exact: true }).click();
  assert.equal(await page.locator("#submit-report").isDisabled(), true);
  assert.equal(await page.locator("#predict-only").isDisabled(), true);
  assert.equal(await page.locator("#screening-form").getAttribute("aria-busy"), "true");
  release();
  await page.waitForFunction(() => !document.getElementById("submit-report").disabled);
  assert.match(await page.locator("#form-errors").innerText(), /saved model is unavailable/);
  assert.equal(new URL(page.url()).hash, "#intake");
  await page.unroute("**/screening-report");
  await page.route("**/predict", (route) => route.abort());
  await page.getByRole("button", { name: "Run model only", exact: true }).click();
  await page.waitForFunction(() => document.getElementById("form-errors").textContent.includes("could not be reached"));
  assert.equal(await page.getByLabel("Participant name", { exact: true }).inputValue(), "Sample Participant");
  await page.goto(`${base}/#summary`);
  assert.match(await page.locator("#live-summary").innerText(), /No live result yet/);
  await context.close();
});

test("422 and malformed success responses are safe and actionable", async () => {
  const { page, context } = await pageFor();
  await page.goto(`${base}/#intake`);
  await page.route("**/predict", (route) => route.fulfill({ status: 422, contentType: "application/json", body: JSON.stringify({ detail: [{ loc: ["body", "age"], msg: "Adult age required" }] }) }));
  await page.getByRole("button", { name: "Run model only", exact: true }).click();
  await page.waitForFunction(() => !document.getElementById("predict-only").disabled);
  assert.equal(await page.evaluate(() => document.activeElement.id), "age");
  assert.equal(await page.locator("#age").getAttribute("aria-invalid"), "true");
  await page.unroute("**/predict");
  await page.route("**/predict", (route) => route.fulfill({ status: 200, contentType: "application/json", body: "{}" }));
  await page.getByRole("button", { name: "Run model only", exact: true }).click();
  await page.waitForFunction(() => document.getElementById("form-errors").textContent.includes("unexpected response"));
  await context.close();
});

test("unavailable evidence offers retry without invented metrics", async () => {
  const { page, context } = await pageFor();
  await page.route("**/research-evidence", (route) => route.fulfill({ status: 503, contentType: "application/json", body: "{}" }));
  await page.goto(`${base}/#research`);
  await page.waitForFunction(() => !document.getElementById("retry-evidence").hidden);
  assert.equal(await page.locator("#research-results table").count(), 0);
  await page.unroute("**/research-evidence");
  await page.getByRole("button", { name: "Retry loading evidence" }).click();
  await page.waitForFunction(() => document.getElementById("evidence-status").textContent.includes("File hashes verified"));
  assert.equal(await page.locator("#research-results table").count(), 1);
  await context.close();
});

test("contradictory API rule label fails visibly instead of being silently overwritten", async () => {
  const { page, context } = await pageFor();
  await page.goto(`${base}/#intake`);
  await page.route("**/predict", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ asd_traits_detected: false, dataset_label: "NO", sum_score: 0, observed_rule_label: "YES", probability: 0.1 }) }));
  await page.getByRole("button", { name: "Run model only", exact: true }).click();
  await page.waitForFunction(() => document.getElementById("form-errors").textContent.includes("unexpected response"));
  assert.equal(new URL(page.url()).hash, "#intake");
  await context.close();
});

test("mobile validation opens hidden invalid fields and associates errors", async () => {
  const { page, context } = await pageFor(390);
  await page.goto(`${base}/#intake`);
  await page.locator("#dataset-fields > summary").click();
  await page.getByLabel("Country of residence", { exact: true }).fill("   ");
  await page.locator("#dataset-fields > summary").click();
  await page.getByRole("button", { name: "Run model only", exact: true }).click();
  assert.equal(await page.locator("#dataset-fields").getAttribute("open"), "");
  assert.equal(await page.evaluate(() => document.activeElement.id), "country_of_res");
  assert.match(await page.locator("#country_of_res").getAttribute("aria-describedby"), /field-error-/);
  await context.close();
});

test("responsive intake and summary at 390 and 320px keep all API fields accessible", async () => {
  for (const width of [390, 320]) {
    const { page, context } = await pageFor(width);
    await page.goto(`${base}/#intake`);
    assert.equal(await overflow(page), false);
    assert.equal(await page.locator("#aq-fields input").count(), 10);
    if (width === 390) await page.screenshot({ path: path.join(screenshotDir, "intake-mobile.png"), fullPage: true });
    await page.locator("#dataset-fields > summary").click();
    assert.equal(await page.getByLabel("Country of residence", { exact: true }).isVisible(), true);
    await page.locator("#dataset-fields > summary").click();
    await page.getByRole("button", { name: "Generate demo summary", exact: true }).click();
    await page.waitForURL("**/#summary");
    assert.equal(await overflow(page), false);
    if (width === 390) await page.screenshot({ path: path.join(screenshotDir, "summary-mobile.png"), fullPage: true });
    await context.close();
  }
  await fs.writeFile(path.join(screenshotDir, "validation.json"), JSON.stringify({ browser: browser.version(), playwright: require("playwright/package.json").version, font: "@fontsource/libre-baskerville 5.3.0 (self-hosted)", desktop: 1200, mobileWidths: [390, 320], evidenceCommit: "ace0186c2ee3c6a8f40d0b20803acc0a59be84c7", scope: "Synthetic fictional inputs; Chromium checks are not screen-reader or real-device certification." }, null, 2) + "\n");
});
