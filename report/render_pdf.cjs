// Print the report HTML to PDF with headless Chromium (Playwright).
// usage: node report/render_pdf.cjs <report.html> <out.pdf> [screenshot_dir]
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const [htmlPath, pdfPath, shotDir] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1056, height: 816 } });
  await page.goto("file://" + path.resolve(htmlPath), { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  await page.pdf({
    path: pdfPath,
    width: "11in",
    height: "8.5in",
    printBackground: true,
    preferCSSPageSize: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
  });
  if (shotDir) {
    // page-by-page PNGs for visual QA
    await page.emulateMedia({ media: "print" });
    const pages = await page.$$("section.page");
    for (let i = 0; i < pages.length; i++) {
      await pages[i].screenshot({ path: path.join(shotDir, `page-${String(i + 1).padStart(2, "0")}.png`) });
    }
  }
  await browser.close();
})();
