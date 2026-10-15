// Print a report/deck HTML to PDF with headless Chromium (Playwright).
// usage: node report/render_pdf.cjs <page.html> <out.pdf> [screenshot_dir] [width] [height]
// width/height default to US Letter landscape (11in x 8.5in); pass "" to skip screenshots.
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const [htmlPath, pdfPath, shotDir, width = "11in", height = "8.5in"] = process.argv.slice(2);
  const px = (v) => Math.round(parseFloat(v) * 96);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: px(width), height: px(height) } });
  await page.goto("file://" + path.resolve(htmlPath), { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  await page.pdf({
    path: pdfPath,
    width,
    height,
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
