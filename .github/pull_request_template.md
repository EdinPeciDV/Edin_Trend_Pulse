## What changed

<!-- One or two sentences. -->

## Type

- [ ] Feature / UI
- [ ] Bug fix
- [ ] Indicator or signal logic
- [ ] Model retrain
- [ ] Infrastructure / CI

## Checks

- [ ] `npm test` passes
- [ ] `npm run build` passes
- [ ] Deploy preview looks right on mobile as well as desktop

### If you touched indicators or features

- [ ] `python3 ml/parity.py` passes — Python and JS still agree
- [ ] Retrained afterwards, or confirmed no retrain is needed

Changing a feature in `shared/mlFeatures.js` without changing
`ml/features.py` (or vice versa) does not fail loudly. It silently feeds
the model out-of-distribution inputs. The parity test is the only thing
that catches it.

### If this is a model change

- [ ] Walk-forward `EDGE` is positive and > ~2x its own SD
- [ ] Positive in at least 4 of 5 folds
- [ ] `net bps` positive after costs, or documented as not tradeable
- [ ] Beats the `heuristic` baseline row
- [ ] Feature weights are plausible

## Risk

<!-- What breaks if this is wrong, and how would you notice? -->
