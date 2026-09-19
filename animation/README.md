# BRACE interactive animation

[Open the animation](https://nebulis-lab.com/BRACE/animation/) · [Project website](https://nebulis-lab.com/BRACE/)

A focused walkthrough of the replanning bottleneck, BRACE gating, token budgets,
E-RECAP compression, phase accounting and the paper’s main platform results.
The project website embeds this same app. Source and deployed assets both live
in **NEBULIS-Lab/BRACE**.

## Develop and publish

Use Node.js 22 (the CI version):

```bash
cd animation
npm ci
npm run dev
npm test
npm run build
```

`npm run build` type-checks the app, builds it with Vite, and copies the output to
`../docs/animation/`. Commit the source changes and generated output together.
The existing GitHub Pages deployment serves `docs/`; there is no separate
animation repository, hosting service, or deployment credential.

To check the full site and embedded app locally:

```bash
npx playwright install chromium
npm run test:browser
```

## Interaction and evidence

- Five labeled chapters; Play/Pause, Replay, Previous/Next and a seekable timeline.
- Chapter selection pauses at the key moment. Play replays that chapter’s transformation.
- Gate scenarios expose routine deferral, unsafe override and budgeted admission.
- Select a token to trace its identity into the retained planner input.
- Switch between Habitat, RoboFactory and AirSim on the results scene.
- Light/dark theme follows the project website; reduced-motion settings are respected.
- Method details and source links are available below the workbench.

The token scores, controller inputs and phase durations are deterministic
teaching examples. The platform results in `app/src/data.ts` reproduce Table 1
of the [paper](../docs/static/main.pdf). The simplified gate cases follow
[the controller specification](../docs/CONTROLLER.md). They do not run inference
or simulate a new experiment.

## Source map

- `app/src/App.tsx`: scenes, playback and interaction.
- `app/src/model.ts`: stable token identities, selection, gate cases and timeline.
- `app/src/data.ts`: paper metrics and deterministic token generation.
- `app/src/styles.css`: responsive layout and shared color tokens.
- `tests/`: model invariants and browser integration checks.
- `scripts/publish.mjs`: build handoff to the existing project site.

The animation is covered by the repository’s [MIT license](../LICENSE).
