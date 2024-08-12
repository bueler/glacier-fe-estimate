# glacier-fe-estimate/py/

The codes here explore the coercivity of the surface evolution of glaciers using Stokes dynamics.  They call the [`stokesextrude`](https://github.com/bueler/stokes-extrude) library package.

## requirements

1. Python3 is required.  Then [download and install Firedrake.](https://www.firedrakeproject.org/download.html)

2. Clone and install `stokesextrude` using [pip](https://pypi.org/project/pip/):

    $ git clone https://github.com/bueler/stokes-extrude.git
    $ cd stokes-extrude/
    $ pip install -e .`

## basic usage on short single case

Activate the [Firedrake](https://www.firedrakeproject.org/) venv and then run a shortened single case:

    $ source firedrake/bin/activate
    $ bash show.sh

This run takes several minutes.  It produces a directory `result/` with subdirectories `azero/`, `aneg/`, and `apos/` for the three SMB cases.  Within these subdirectories are time-dependent `.png` states which can be regarded as movies for the SMB cases.

At the end of the simulations, coercivity ratios are sampled from the many stored states.  Image files are written into `result/`: `badcoercive-T1-T2.png` showing cases where the coercivity ratio was negative, `snaps.png` showing a few states, and `Phiratios.png` showing a histogram like that in the paper.

There is also a file `ratios.txt` which includes the basic statistics for the coercivity ratios reported in the paper.

## reproduce the results shown in the paper

The full study takes something like 10 hours.  It does many bed/SMB cases and several resolutions.  It produces subdirectories and files similar to those described above:

    $ cd reproduce/
    $ bash study.sh
