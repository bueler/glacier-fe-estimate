# glacier-fe-estimate/py/

Use the [`stokesextrude`](https://github.com/bueler/stokes-extrude) package to run a study of the glacier geometry bounds described in the paper.

## requirements

1. [Download and install Firedrake.](https://www.firedrakeproject.org/download.html)

2. Clone and install `stokesextrude` using [pip](https://pypi.org/project/pip/):

    $ git clone https://github.com/bueler/stokes-extrude.git
    $ cd stokes-extrude/
    $ pip install -e .`

## basic usage

Activate the [Firedrake](https://www.firedrakeproject.org/) venv and then run the study:

    $ source firedrake/bin/activate
    $ python3 study.py
