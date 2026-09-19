import { cp, mkdir, readdir, rm } from 'node:fs/promises';
const source = new URL('../dist/', import.meta.url);
const target = new URL('../../docs/animation/', import.meta.url);
await mkdir(target, { recursive: true });
// Only replace generated app assets; keep the README preview image.
for (const entry of await readdir(target)) {
  if (entry !== 'preview.png') await rm(new URL(entry, target), { recursive: true, force: true });
}
await cp(source, target, { recursive: true });
console.log('Published animation build to docs/animation/');
