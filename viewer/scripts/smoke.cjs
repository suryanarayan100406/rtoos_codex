/* Real built-UI smoke only, with an empty run root; not reconstruction acceptance.
   Use an existing local Playwright installation, no browser/runtime downloads.
   DRISHTI_PYTHON=/path/to/python PLAYWRIGHT_MODULE=/path/to/playwright node scripts/smoke.cjs */
const { spawn } = require("node:child_process");
const { resolve } = require("node:path");
const net = require("node:net");
const assert = require("node:assert/strict");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

(async () => {
  const socket = net.createServer();
  await new Promise(resolve => socket.listen(0, "127.0.0.1", resolve));
  const port = socket.address().port;
  await new Promise(resolve => socket.close(resolve));
  const root = resolve(__dirname, "../..");
  const server = spawn(process.env.DRISHTI_PYTHON || "python", ["-m", "server.app"], {
    cwd: root, env: { ...process.env, DRISHTI_SERVER_PORT: String(port),
      DRISHTI_SERVER_RUNS_ROOT: "/tmp/omnirush/drishti-ui-empty-run-root" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let log = "";
  server.stdout.on("data", chunk => { log += chunk; });
  server.stderr.on("data", chunk => { log += chunk; });
  let browser;
  try {
    const base = `http://127.0.0.1:${port}`;
    for (let attempt = 0; ; attempt++) {
      try { const response = await fetch(`${base}/api/health`); assert(response.ok); break; }
      catch (error) { if (attempt > 100) throw new Error(`Local module failed to start: ${log}`); }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    const runs = await (await fetch(`${base}/api/runs`)).json();
    assert.equal(runs.runs.length, 0, "The smoke root must be empty");
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const failures = [], external = [];
    page.on("pageerror", error => failures.push(error.message));
    page.on("request", request => {
      if (!request.url().startsWith(base) && !request.url().startsWith("data:") && !request.url().startsWith("blob:")) {
        external.push(request.url());
      }
    });
    await page.goto(base, { waitUntil: "networkidle" });
    await page.getByRole("heading", { name: "No reconstruction loaded" }).waitFor();
    await page.getByRole("button", { name: "+ New run", exact: true }).click();
    await page.getByRole("heading", { name: "Start a recorded-data run" }).waitFor();
    await page.getByRole("button", { name: "Close", exact: true }).click();
    await page.getByRole("button", { name: "Inspect runtime / doctor" }).click();
    await page.getByRole("heading", { name: "Detected local environment" }).waitFor();
    await page.getByRole("button", { name: "Close", exact: true }).click();
    assert.deepEqual(failures, []);
    assert.deepEqual(external, []);
    await page.setViewportSize({ width: 390, height: 844 });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
    assert.equal(overflow, false, "Mobile console must fit the viewport");
    console.log("PASS: executable module, built console, empty state, run dialog, doctor, mobile width, no page errors or external requests");
  } finally {
    if (browser) await browser.close();
    server.kill("SIGTERM");
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
