# Wobble board mechanics, from base shape to stability: code

Simulation, numerical, and plotting scripts for the paper

> T. Deligiannis, B. Schlattmann, J. Fabianiak, M. Mangalam.
> *Wobble board mechanics, from base shape to stability.* (manuscript, 2026)

The scripts reproduce the figures, Table 1, the worked examples, and the verification checks
reported in the Methods. Nothing here uses experimental data.

## Requirements

Python 3.10 or later with `numpy`, `scipy`, and `matplotlib` (`pip install -r requirements.txt`).
Run each script from its own folder, for example `cd figures && python make_torque_figures.py`.
Figures are written to `figures/output/`; some simulation scripts write a `.json` file of results
next to themselves.

## Contents

### `numbers/`

| Script | Produces | Paper |
|---|---|---|
| `board_numbers.py` | Critical aspect ratio with a platform; Table 1; edge-contact angles; the laboratory boards in the light-base limit; boards S and U; the loaded-board estimate | Sec. 2.4, 2.5, 3.3, 4.1; Table 1 |

### `figures/`

| Script | Produces | Paper |
|---|---|---|
| `make_board_classification.py` | `board_classification.pdf/.png` | Fig. 1 |
| `make_support_function.py` | `support_function_full.pdf/.png` | Fig. 2 |
| `make_torque_figures.py` | `torque_field.pdf/.png`, `torque_trajectory.pdf/.png` | Figs. 3 and 4 |
| `boardgeom.py` | Shared geometry: solid half-ellipsoidal base with a disk platform | Eq. (platform) |

Figs. 3 and 4 are for the unloaded board with its platform (45 cm x 2 cm disk, base of
b = 150 mm, mass ratio estimated for a base and platform of the same material). Fig. 4b also
shows the base alone.

### `simulation/`

`board3d.py` is the three-dimensional model: a rigid solid half-ellipsoid rolling without slip
on a horizontal floor, with all three rotations free, quaternion orientation, and DOP853
integration (rtol 1e-10, atol 1e-12, maximum step 2 ms). It simulates the base alone, b = 0.15 m.
In the code, `rho` is the vector from the centroid to the contact point, as in the Methods.

| Script | Checks | Expected output |
|---|---|---|
| `energy_check.py` | Energy conservation, triaxial base, 5 s | relative drift 2e-12 |
| `exact_eom_check.py` | 3D solution against the exact planar equation | 2.1e-6, 9.7e-8, 2.7e-7 deg |
| `validate.py` | Sphere against the exact planar equation; energy for three releases | 1.3e-9 deg; about 1e-10 of the initial excess energy |
| `caseA.py` | In-plane releases from 20 deg: time to level, zero yaw | 0.179, 0.262, 0.932 s |
| `caseB.py` | Stable triaxial bases released off-axis: bounded yaw | Sec. 2.5 |
| `caseC2.py` | Releases while circling, no axial spin: yaw accumulation | Sec. 2.5 |
| `phase.py` | Yaw equals the solid angle swept by the board axis | Eq. (yaw_phase) |
| `spin_check.py` | Yaw with and without spin about the board axis | Sec. 2.5 |

The longer simulations (`caseB.py`, `caseC2.py`, `phase.py`, `spin_check.py`) take a few minutes each.

## Model summary

- Base: homogeneous solid ellipsoid with semi-axes a1 (AP), a2 (ML), b (vertical), cut by a
  horizontal plane; aspect ratios rho = b/a.
- Platform: uniform disk of diameter w and thickness t on the cut face, mass ratio mu = m_p/m_b.
- Signed stiffness per unit weight: q0 - h0, with q0 = a^2/b the radius of curvature at the
  contact and h0 the centroid height of the whole board. The board is stable in a direction
  when q0 > h0.

## Citing

If you use this code, please cite the paper and this repository (see `CITATION.cff`).

## License

MIT, see `LICENSE`.
