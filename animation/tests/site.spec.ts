import { test, expect } from '@playwright/test';

test('the original seven-step animation supports next, back, direct navigation and restart', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('animation/');
  await expect(page.locator('.step-dots button')).toHaveCount(7);
  await expect(page.locator('.progress-cluster strong')).toHaveText('Context');
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(page.locator('.progress-cluster strong')).toHaveText('Pressure');
  await page.getByRole('button', { name: 'Back', exact: true }).click();
  await expect(page.locator('.progress-cluster strong')).toHaveText('Context');
  for (const label of ['Gate', 'Budget', 'E-RECAP', 'Audit', 'Evidence']) {
    await page.getByRole('button', { name: `Go to ${label}`, exact: true }).click();
    await expect(page.locator('.progress-cluster strong')).toHaveText(label);
  }
  await page.getByRole('button', { name: 'Restart', exact: true }).click();
  await expect(page.locator('.progress-cluster strong')).toHaveText('Context');
  expect(errors).toEqual([]);
});

test('homepage only loads a static preview and opens the animation on click', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', request => requests.push(request.url()));
  await page.goto('');
  await page.locator('#animation').scrollIntoViewIfNeeded();
  await expect(page.locator('iframe, video')).toHaveCount(0);
  await expect(page.locator('#animation')).toHaveAttribute('href', 'animation/');
  await expect.poll(() => page.locator('#animation img').evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth > 0)).toBe(true);
  expect(requests.some(url => /\/animation\/(assets\/|\?embed)|youtube|shuaijun-liu.github.io/i.test(url))).toBe(false);
  await page.locator('#animation').click();
  await expect(page).toHaveURL(/\/BRACE\/animation\/$/);
  await expect(page.locator('.step-dots button')).toHaveCount(7);
});

test('homepage preview remains compact and readable on desktop and mobile in both themes', async ({ page }, testInfo) => {
  await page.goto('');
  for (const width of [1366, 390]) {
    await page.setViewportSize({ width, height: 900 });
    for (const dark of [false, true]) {
      if ((await page.locator('html').getAttribute('data-theme') === 'dark') !== dark) await page.getByRole('button', { name: /Switch to .* theme/ }).click();
      await page.locator('#animation').scrollIntoViewIfNeeded();
      const box = await page.locator('#animation').boundingBox();
      expect(box!.height).toBeLessThan(450);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.locator('#animation').screenshot({ path: testInfo.outputPath(`preview-${width}-${dark ? 'dark' : 'light'}.png`) });
    }
  }
});
