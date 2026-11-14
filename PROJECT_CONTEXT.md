# PlanetHunter — Project Context and Handoff

## Deadline

November 15, 2026, midnight.

## Objective

Build a physics-informed machine learning prototype for detecting and ranking moving astronomical objects using multi-epoch NASA SPHEREx observations, with particular relevance to distant solar-system bodies and Planet Nine searches.

The project must demonstrate a scientifically credible search methodology, not claim discovery of a real planet.

## Repository

https://github.com/dishitagarwal22-cyber/PlanetHunter

Local directory: `~/PlanetHunter`

Environment: macOS, Python virtual environment `.venv`.

Run scripts from the project root.

## Current Dataset

Four registered SPHEREx observations around RA 337.8783221 degrees, Dec -5.1891588 degrees.

- Epoch 1: 2025-05-23
- Epoch 2: 2025-06-19
- Epoch 3: 2025-11-20
- Epoch 4: 2026-05-21

Field size: approximately 0.1 degrees.

Pixel scale: approximately 6.1 arcseconds per pixel.

Original FITS files contain IMAGE, FLAGS, VARIANCE, ZODI, EPSF and WCS-WAVE extensions.

Registered FITS files contain the intensity image and reference WCS, but do not preserve all original scientific extensions.

## Existing Core Scripts

- `src/search.py`: SPHEREx archive search.
- `src/multi_epoch.py`: Download and register observations.
- `src/view_epochs.py`: Visualize aligned observations.
- `src/source_match.py`: Original two-epoch footprint-overlap matching.
- `src/inject.py`: Original synthetic moving-source injection.
- `src/source_detection.py`: Reusable MAD-based source detection and intensity-weighted centroid calculation.
- `src/source_catalogue.py`: Generate multi-epoch detection catalogue and footprints.
- `src/stationary_association.py`: Pairwise stationary associations using spherical angular separations.
- `src/stationary_tracks.py`: Union-Find grouping of stationary associations, with consistency checks.

Earlier experiments are preserved in `experiments/`. Scientific diagnostic scripts are in `tools/`.

## Current Results

Source detections per epoch: 14, 23, 20, 14.

Total detections: 71.

Stationary pairwise associations: 82, all mutual nearest neighbours.

Median separation: 0.807 arcseconds. This is positional repeatability of selected associations, not calibrated absolute astrometric accuracy.

Stationary grouping results:

- 25 total source groups.
- 11 groups with four epochs.
- 4 groups with three epochs.
- 5 groups with two epochs.
- 5 groups with one epoch.

Classification:

- 14 stationary candidates.
- 5 two-epoch candidates.
- 5 unassociated detections.
- 1 inconsistent group.
- 0 conflicting groups.

The inconsistent group S008 is worth investigating but may be caused by blending or centroid bias. It is not a confirmed moving object.

## Previous Synthetic Experiment

Injected a moving Gaussian source into two real astronomical images.

True positions: (15, 20) and (35, 30).

Recovered positions: (15.03, 20.03) and (34.98, 30.00).

True displacement: 22.36 pixels.

Recovered displacement: 22.31 pixels.

Correct association ranked first in the preliminary two-epoch experiment.

## Important Data Files

- `data/observations.csv`
- `data/epochs.csv`
- `data/epochs/`
- `data/source_catalogue.csv`
- `data/source_footprints.npz`
- `data/source_catalogue_preview.png`
- `data/stationary_pairs.csv`
- `data/stationary_tracks.csv`
- `data/track_members.csv`
- `data/stationary_tracks.png`

## Next Development Tasks

1. Build moving-object trajectory association using detections from three or more epochs.
2. Implement stationary, constant-velocity and parallax-aware motion models.
3. Generate physically plausible synthetic moving objects across multiple epochs.
4. Run source detection and trajectory association on injected images.
5. Generate labelled correct and incorrect tracks.
6. Train a Random Forest classifier to rank candidate tracks.
7. Compare ML against a physics-only baseline using held-out simulations.
8. Build a Streamlit dashboard.
9. Prepare final submission materials.

## Scientific Requirements and Limitations

- Do not permanently discard stationary candidates; slow-moving objects may appear stationary.
- Do not assume constant brightness or identical wavelength coverage.
- Missing detections must be supported.
- A source appearing or disappearing does not establish motion.
- Use actual observation timestamps.
- Parallax modelling should use observer geometry; Earth-centre ephemerides can serve as an explicitly labelled approximation if spacecraft ephemerides are unavailable.
- The current field is too small to contain every plausible distant-planet trajectory. A 500 AU object can have an annual parallax amplitude of approximately 413 arcseconds, exceeding the field width.
- Empirical peak SNR is not a calibrated photometric SNR.
- Source morphology measurements are not centroid uncertainties.
- Synthetic injection recovery does not prove real-world discovery capability.
- Avoid overfitting and training/test leakage.
- Compare ML performance against a non-ML baseline.

## Development Preferences

The project owner understands astronomy and mathematics but is new to machine learning. Explain the scientific purpose of each major component before implementing it.

Prioritize a complete working hackathon prototype over unnecessary complexity. Preserve working code and use Git commits at milestones.

## Current Stopping Point

Stationary-source grouping is complete and working. The next task is moving-object trajectory generation and physical model fitting. No multi-epoch moving-object engine or ML model has been implemented yet.
