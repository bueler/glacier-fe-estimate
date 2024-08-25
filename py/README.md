# glacier-fe-estimate/py/

The codes here explore the coercivity of the surface evolution of glaciers using Stokes dynamics.  They call the [`stokes-extrude`](https://github.com/bueler/stokes-extrude) library package.

## requirements

1. Python3 is required.  Then [download and install Firedrake.](https://www.firedrakeproject.org/download.html)

2. Clone and install `stokes-extrude` using [pip](https://pypi.org/project/pip/):

```bash
$ git clone https://github.com/bueler/stokes-extrude.git
$ cd stokes-extrude/
$ pip install -e .
```

## basic usage on short single case

Return to the current directory `glacier-fe-estimate/py/`.  Activate the [Firedrake](https://www.firedrakeproject.org/) venv and then run a shortened single case:

```bash
$ source firedrake/bin/activate
$ bash show.sh
```

This run takes several minutes.  It produces a directory `result/` with subdirectories `azero/`, `aneg/`, and `apos/` for the three SMB cases.  Within these subdirectories are time-dependent `.png` states which can be regarded as movies for the SMB cases.

At the end of the simulations, coercivity ratios are sampled from the many stored states.  Image files are written into `result/`: `badcoercive-T1-T2.png` showing cases where the coercivity ratio was negative, `snaps.png` showing a few states, and `Phiratios.png` showing a histogram like that in the paper.

There is also a file `ratios.txt` which includes the basic statistics for the coercivity ratios reported in the paper.

## reproduce the results shown in the paper

The full study takes something like 10 hours.  It does many bed/SMB cases and several resolutions.  It produces subdirectories and files similar to those described above:

```bash
$ cd reproduce/
$ bash study.sh
```

## how it works

  * The code in `case.py` runs one case as specified by runtime options.  One case fixes the bed type (i.e. flat, smooth, or rough) and fixes the resolution, but it includes three restarts with different values of SMB.  In all cases the 2D glacier has initial Halfar profile over a chosen bed.
  * Running a case does time-steps using the free-surface stabilization algorithm (FSSA) from Lofgren et al 2022.  (This is a change in the Stokes solve.)  The time-stepping does semi-implicit solves for the VI problem arising from the backward Euler time step for the surface kinematical equation NCP.
  * Optional explicit steps are available.
  * Each time step computes and saves the surface elevation s, surface velocity u|_s, and the surface motion map Phi(s) = - u|_s . n_s for evaluation.
  * The evaluation stage at the end, `sampleratios()`, computes ratios between random state pairs to evaluate Conjectures A and B.
  * This runs only in serial.
  * The details are documented in the paper.
  * See `show.sh` and `reproduce/study.sh` for how to run one case.
  * To write optional t-dependent image files into directory do: `python3 case.py 201 15 20 1.0 flat ratios.txt result/` or similar.  This writes `result/azero/*.png`, `result/aneg/*.png`, `result/apos/*.png`
  * To write an optional t-dependent `.pvd` file with Stokes results and diagnostics, also append a filename root: `python3 case.py 201 15 20 1.0 flat ratios.txt result/ result` or similar.  This writes `result_azero.pvd`, `result_aneg.pvd`, `result_apos.pvd`.
