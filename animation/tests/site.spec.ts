import { test, expect } from '@playwright/test';

async function noOverflow(page: import('@playwright/test').Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
}

test('chapters expose the gate decision and trace stable retained tokens', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('animation/');
  await expect(page.locator('h1')).toContainText('Good plans');
  await page.getByRole('button', { name: '02 When to call' }).click();
  await expect(page.locator('.gate-output')).toContainText('Keep executing');
  await expect(page.locator('.call-count b')).toHaveText('0');
  await page.getByRole('button', { name: 'Unsafe state', exact: true }).click();
  await expect(page.locator('.gate-output')).toContainText('Admit recovery');
  await expect(page.locator('.call-count b')).toHaveText('1');
  await page.getByRole('button', { name: 'Windows cleared' }).click();
  await expect(page.locator('.gate-output')).toContainText('Admit + budget');
  await page.getByRole('button', { name: '03 How much to spend' }).click();
  await expect(page.locator('.token')).toHaveCount(32);
  await expect(page.locator('.token.kept')).toHaveCount(12);
  await expect(page.locator('.token.protected.kept')).toHaveCount(6);
  const ids = await page.locator('.token').evaluateAll(els => els.map(el => el.getAttribute('data-token-id')));
  await page.locator('.token.kept').first().click();
  await expect(page.locator('.token-inspector')).toContainText('retained in its original order');
  await expect(page.locator('.packed-tokens .inspected')).toHaveCount(1);
  await page.getByRole('button', { name: 'Previous chapter' }).click();
  await page.getByRole('button', { name: 'Next chapter' }).click();
  expect(await page.locator('.token').evaluateAll(els => els.map(el => el.getAttribute('data-token-id')))).toEqual(ids);
  expect(errors).toEqual([]);
});

test('playback pauses, seeks backward, replays and finishes the full tour', async ({ page }) => {
  await page.goto('animation/');
  await page.getByRole('button', { name: 'Play tour' }).click();
  await page.waitForTimeout(1100);
  await page.getByRole('button', { name: 'Pause', exact: false }).click();
  const position = await page.getByRole('slider').inputValue();
  await page.waitForTimeout(400);
  expect(await page.getByRole('slider').inputValue()).toEqual(position);
  await page.getByRole('slider').fill('10');
  await page.getByRole('button', { name: 'Play tour' }).click();
  await page.waitForTimeout(300);
  await page.getByRole('button', { name: 'Pause', exact: false }).click();
  expect(Number(await page.getByRole('slider').inputValue())).toBeGreaterThan(10);
  await page.getByRole('slider').fill('45');
  await expect(page.locator('h1')).toContainText('entire call');
  await page.getByRole('slider').fill('30');
  await expect(page.locator('.token')).toHaveCount(32);
  await page.getByRole('slider').fill('60');
  await expect(page.getByRole('button', { name: 'Replay', exact: false }).first()).toBeVisible();
  await page.getByRole('button', { name: 'Replay from beginning' }).click();
  await expect(page.locator('h1')).toContainText('Good plans');
  await expect(page.getByRole('slider')).toHaveValue('60', { timeout: 70_000 });
  await expect(page.locator('h1')).toContainText('Fewer deadline');
  await expect(page.getByRole('button', { name: 'Pause', exact: false })).toHaveCount(0);
});

test('reported outcomes remain platform specific', async ({ page }) => {
  await page.goto('animation/');
  await page.getByRole('button', { name: '05 What changes' }).click();
  await expect(page.locator('.comparison-row').last()).toContainText('4.7%');
  await page.getByRole('button', { name: 'RoboFactory', exact: true }).click();
  await expect(page.locator('.comparison-row').last()).toContainText('50%');
  await expect(page.locator('.evidence-context')).toContainText('250 ms');
  await expect(page.locator('.evidence-metrics')).toContainText('1,566');
  await page.getByRole('button', { name: 'Habitat', exact: true }).click();
  await expect(page.locator('.comparison-row').last()).toContainText('4.7%');
  await expect(page.locator('.evidence-metrics')).toContainText('235');
});

test('all chapters fit desktop, tablet and mobile in both themes', async ({ page }, testInfo) => {
  await page.goto('animation/');
  for (const width of [1366, 768, 390]) {
    await page.setViewportSize({ width, height: 900 });
    for (const dark of [false, true]) {
      if ((await page.locator('html').getAttribute('data-theme') === 'dark') !== dark) await page.getByRole('button', { name: /Switch to .* theme/ }).click();
      for (let index = 0; index < 5; index++) {
        await page.locator('.chapters button').nth(index).click();
        await noOverflow(page);
        const controls = await page.locator('.playback').boundingBox();
        if (width === 1366) expect(controls!.y + controls!.height).toBeLessThan(900);
        if (width !== 768 && [0, 2, 4].includes(index)) await page.screenshot({ path: testInfo.outputPath(`${width}-${dark ? 'dark' : 'light'}-chapter-${index}.png`), fullPage: true });
      }
    }
  }
});

test('the project embeds its own app, resizes it and shares theme', async ({ page }, testInfo) => {
  const errors: string[] = [], requests: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('request', r => requests.push(r.url()));
  await page.goto('');
  await page.locator('#animation').scrollIntoViewIfNeeded();
  const app = page.frameLocator('#brace-animation');
  await expect(app.locator('h1')).toContainText('Good plans');
  const frameUrl = await page.locator('#brace-animation').getAttribute('src');
  expect(frameUrl).toBe('animation/?embed=1');
  await expect(page.locator('video, [data-video-src], iframe[src*="youtube"]')).toHaveCount(0);
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await expect(app.locator('html')).toHaveAttribute('data-theme', 'dark');
  await app.getByRole('button', { name: 'Method & sources +' }).click();
  await expect(app.locator('#method-details')).toBeVisible();
  await expect.poll(async () => {
    const height = await page.locator('#brace-animation').evaluate(el => el.getBoundingClientRect().height);
    const contentHeight = await app.locator('.app-shell').evaluate(el => el.getBoundingClientRect().height);
    return height >= contentHeight;
  }).toBe(true);
  await app.getByRole('button', { name: 'Hide details −' }).click();
  await app.getByRole('button', { name: '03 How much to spend' }).click();
  await page.locator('#animation').screenshot({ path: testInfo.outputPath('embedded-desktop.png') });
  await page.setViewportSize({ width: 390, height: 844 });
  await noOverflow(page);
  expect(await app.locator('body').evaluate(el => el.scrollWidth <= innerWidth)).toBe(true);
  await expect(page.locator('.scroll-to-top')).toBeHidden();
  await expect(page.locator('.theme-control')).toBeHidden();
  await app.getByRole('button', { name: 'Switch to light theme' }).click();
  await expect(page.locator('html')).not.toHaveAttribute('data-theme', 'dark');
  await page.locator('#animation').screenshot({ path: testInfo.outputPath('embedded-mobile.png') });
  expect(requests.some(url => /youtube|shuaijun-liu.github.io/i.test(url))).toBe(false);
  expect(errors).toEqual([]);
});

test('reduced motion retains readable outcomes and keyboard navigation', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('animation/');
  await page.locator('h1').click();
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('h1')).toContainText('trigger');
  await page.keyboard.press('Space');
  await expect(page.getByRole('button', { name: 'Pause', exact: false })).toBeVisible();
  await page.keyboard.press('Space');
  await page.getByRole('button', { name: '03 How much to spend' }).click();
  await expect(page.locator('.token.kept')).toHaveCount(12);
});
