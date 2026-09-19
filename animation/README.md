# BRACE interactive animation

[Open the animation](https://nebulis-lab.com/BRACE/animation/) · [Project website](https://nebulis-lab.com/BRACE/)

The original guided BRACE animation, maintained in **NEBULIS-Lab/BRACE**.
Its seven steps cover context growth, trigger pressure, the stability gate,
budget assignment, E-RECAP compression, phase accounting and cross-platform evidence.
Use **Next**, **Back**, or the step navigation to explore the walkthrough.

The project homepage shows a small static preview linking to the dedicated
animation page. The interactive app runs only after opening that page.

## Development and publishing

Use Node.js 22:

```bash
cd animation
npm ci
npm run dev
npm test
npm run build
```

`npm test` type-checks the source. `npm run build` builds the app and copies the
output into `../docs/animation/`. Commit source and generated output together;
the existing GitHub Pages deployment serves `docs/`.

To verify the project-page entry and original animation navigation:

```bash
npx playwright install chromium
npm run test:browser
```

The source is under `app/src/`; `scripts/publish.mjs` maintains the generated
site assets while preserving the static `docs/animation/preview.png` thumbnail.
The app uses relative asset paths and works under the project’s `/BRACE/animation/`
URL. No separate animation repository or deployment service is required.

The animation is covered by the repository’s [MIT license](../LICENSE).
