// 대시보드 탭 멈춤 시험 — 모든 단계에 시간제한. 화면이 멈추면 기다리지 않고 FREEZE 로 판정한다.
// (RETRO_텍스트정규화 오류 53: MCP 브라우저로 시험하다 멈춘 페이지에 묶여 작업이 반복 중단됨)
//
// 사용 (Git Bash, 시험 서버를 먼저 띄운 뒤):
//   NODE_PATH="$(npm root -g)" timeout 110 node tests/ui_tab_check.js http://127.0.0.1:7871 [시험.pdf]
//   EDGE=1 ... → 실제 Edge 창으로 실행
// pdf 를 주면 '전체 처리' 후 ③ 탭 표에 그 파일 기록이 보이는지까지 확인한다.
// 시험 pdf 는 텍스트 레이어 60자 이상(pdf_text_layer.MIN_DOC_CHARS) — 모자라면 스캔본으로
// 판정돼 Vision API 가 호출된다.
// 시험 기록은 끝난 뒤 storage_manager.delete_entry(id, delete_files=True) 와
// 02_결과·storage/results 의 *_원문.txt 로 지운다.
const { chromium } = require('playwright');
const [url, pdf] = process.argv.slice(2);
const launchOpts = process.env.EDGE ? { channel: 'msedge', headless: false } : { headless: true };

const alive = (page, ms = 5000) => Promise.race([
  page.evaluate(() => 1).then(() => true),
  new Promise(r => setTimeout(() => r(false), ms)),
]);

async function openTab(page, mark) {
  await page.locator('button[role=tab]', { hasText: mark }).click({ timeout: 5000, noWaitAfter: true }).catch(() => {});
  return alive(page);
}

// 문장 정제·L3 체크박스를 끄고, 꺼졌는지 돌려준다(시험 비용 0 보장)
const aiOff = page => page.evaluate(() => {
  const boxes = [...document.querySelectorAll('label')]
    .filter(l => l.textContent.includes('문장 정제') || l.textContent.includes('L3'))
    .map(l => l.querySelector('input[type=checkbox]')).filter(Boolean);
  boxes.forEach(cb => { if (cb.checked) cb.click(); });
  return boxes.every(cb => !cb.checked);
});

(async () => {
  const browser = await chromium.launch(launchOpts);
  const out = {};
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    await page.goto(url, { timeout: 20000 });
    await page.waitForSelector('button[role=tab]', { timeout: 20000 });
    await page.waitForTimeout(1500);  // 화면 연결이 끝난 뒤 체크박스를 만진다

    if (pdf) {
      await page.setInputFiles('input[type=file]', pdf, { timeout: 10000 });
      await page.waitForTimeout(2000);
      out.ai_off = (await aiOff(page)) && (await page.waitForTimeout(500), await aiOff(page));
      if (!out.ai_off) throw new Error('AI 체크박스를 끄지 못해 처리를 건너뜀');
      await page.getByRole('button', { name: '전체 처리' }).click({ timeout: 5000 });
      await page.getByText(/완료\s+·\s+총/).first().waitFor({ timeout: 60000 });
      await page.waitForTimeout(1000);
      out.processed = true;
    }

    out.tab2 = (await openTab(page, '②')) ? 'OK' : 'FREEZE';
    if (out.tab2 === 'OK') out.tab3 = (await openTab(page, '③')) ? 'OK' : 'FREEZE';
    if (out.tab3 === 'OK') {
      await page.waitForTimeout(1500);  // 탭 select 갱신 응답 대기
      out.tab3_after_refresh = (await alive(page)) ? 'OK' : 'FREEZE';
      out.stats_visible = (await page.evaluate(() => document.body.innerText)).includes('저장 현황');
      if (pdf) {
        const name = pdf.split(/[\\/]/).pop().replace(/\.[^.]+$/, '');
        out.record_in_tab3_table = await page.evaluate(n =>
          [...document.querySelectorAll('table')].some(t => t.offsetParent && t.innerText.includes(n)), name);
      }
    }
  } catch (e) {
    out.error = String(e).split('\n')[0];
  }
  console.log(JSON.stringify(out));
  await Promise.race([browser.close(), new Promise(r => setTimeout(r, 3000))]);
  process.exit(0);
})();
