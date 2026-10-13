const fs = require("node:fs");
const path = require("node:path");

const result = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const filename = process.argv[3] || "CLAUDE_REVIEW.md";
const cliVersion = process.argv[4] || "unrecorded";
if (path.basename(filename) !== filename || !filename.endsWith(".md")) throw new Error("Use a Markdown filename inside docs/ui");
if (result.is_error || result.subtype !== "success" || typeof result.result !== "string") {
  throw new Error("No successful independent review to record");
}
const header = `# Claude Code Review Record\n\nCLI: ${cliVersion}. Runtime models: ${Object.keys(result.modelUsage || {}).join(", ")}.\nAccess: captured Figma reference PNGs, not live Figma MCP. Source edits were disabled.\n\nThe original reviewer output below is preserved verbatim, including hypotheses and\nmisinterpretations. See research/TASKS.md for verified triage and corrections.\nDenied tool calls: ${(result.permission_denials || []).length}. No billing/session metadata is published.\n\n---\n\n`;
fs.writeFileSync(path.resolve(__dirname, "../docs/ui", filename), header + result.result + "\n");
