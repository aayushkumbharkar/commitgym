# Design references and implementation

The requested resources were reviewed during implementation:

- [Taste Skill](https://www.tasteskill.dev/): typography hierarchy, restrained color, disciplined spacing, and avoiding repeated generic card layouts informed the persistent workbench. The dashboard brief takes precedence over landing-page effects. No claim is made that a downloaded skill package was installed.
- [21st.dev shadcn Button](https://21st.dev/@shadcn/components/button): default/secondary/ghost variants, icon-bearing actions and disabled/loading behavior informed the reusable typed Button. The published usage examples were consulted. Component source requires a site account; this repository contains an original compact, shadcn-compatible CVA/ref-forwarding implementation, not a copied account-gated asset.
- [ThreeUI Topology Field](https://threeui.com/three-js/structure-flow/topology-field): connected-node and dependency visualization informed the three-node operational diagram. Its source preview did not load through the available basic page. The implementation is original responsive SVG, not a bundled ThreeUI WebGL component. This keeps the core operational view available without WebGL and uses real state.
- [GSAP](https://gsap.com/): installed as a dependency and used for state revision and timeline transitions. `matchMedia` respects reduced motion; cleanup prevents duplicate tweens. Animations do not postpone API actions.

UI UX Pro Max was not found in the installed skills searched for this session. Its application is not claimed.

The result uses graphite surfaces, acid-green accents, fine borders, small technical labels, a large central workspace and a narrow evidence inspector. At small widths navigation becomes horizontally scrollable and the topology stacks vertically. Actual screenshots are under `screenshots/`. They were captured from the running local app, not generated as design mockups.

Browser QA checks automated accessibility and overflow at 375/768/1440px, along with actual navigation and simulator actions. There was no preexisting screenshot baseline, so visual regression is inconclusive; current screenshots establish a future baseline. Automated axe checks do not replace manual keyboard and screen-reader testing.
