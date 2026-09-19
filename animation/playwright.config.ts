import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.ts',
  timeout: 80_000,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: process.env.BRACE_TEST_URL || 'http://127.0.0.1:4173/BRACE/',
    viewport: { width: 1366, height: 900 },
    launchOptions: {
      executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
      args: ['--disable-gpu', '--disable-gpu-compositing', '--disable-webgl', '--disable-accelerated-2d-canvas'],
    },
  },
  webServer: process.env.BRACE_TEST_URL ? undefined : {
    command: 'node scripts/serve.mjs',
    url: 'http://127.0.0.1:4173/BRACE/',
    reuseExistingServer: !process.env.CI,
  },
});
