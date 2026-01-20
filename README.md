


## Prior Localization – RT Subgroup Extension

This repository is a **extension** of the original
[prior-localization](https://github.com/int-brain-lab/prior-localization)
repository by Findling, Hubert et al

## Original paper:  
Findling, Hubert et al. (2023), *Brain-wide representations of prior information in mouse decision-making*

This fork modifies the original pipeline to perform **reaction-time (RT)–based subgroup analyses**
(e.g. fast / normal / slow trials) while preserving the original decoding framework.

## What is different in this fork?

Compared to the original implementation, this repository adds:

  - RT-based trial stratification**
  - Trials are grouped into `fast`, `normal`, and `slow` based on custom reaction-time cutoffs
  - Reaction time is computed from wheel-velocity–based movement onset detection algotirhm in prior-localization/prior_localization/my_rt.py
  - Decoding is run separately for each RT subgroup
  - Parallel processing across sessions


All original model assumptions, decoding logic, and estimators remain unchanged unless explicitly stated.


## Dependencies
The code has been tested on Ubuntu 20.04 and 22.04, Rocky Linux 8.8 and OSX 13.4.1, using Python 3.8, 3.9 and 3.10.
Required Python software packages are listed in [requirements.txt](https://github.com/int-brain-lab/prior-localization/blob/main/requirements.txt). 

## Installation
The installation takes about 7 min on a standard desktop computer. It is recommended to set up and activate a clean environment using conda or virtualenv, e.g.
```shell
virtualenv prior --python=python3.10
source prior/bin/activate
```

Then clone this repository and install it along with its dependencies
```shell
git clone https://github.com/Kelly-Yin-99/prior-localization-rt.git
cd prior-localization-rt
pip install -e .
```

In a Python console, test if you can import functions from prior_localization
```python
from prior_localization.fit_data import fit_session_ephys
```


## Connecting to IBL database
In order to run the example code or the tests, you need to connect to the public IBL database to access example data.
Our API, the Open Neurophysiology Environment (ONE) has already been installed with the requirements. 
If you have never used ONE, you can just establish the default database connection like this in a Python console. 
The first time you instantiate ONE you will have to enter the password (`international`) 
```python
from one.api import ONE
ONE.setup(silent=True)
one = ONE()
```

**NOTE**: if you have previously used ONE with a different database you might want to run this instead. Again, the 
first time you instantiate ONE you will have to enter the password (`international`)
```python
from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', make_default=False, silent=True)
one = ONE(base_url='https://openalyx.internationalbrainlab.org')
```

If you run into any issues refer to the [ONE documentation](https://int-brain-lab.github.io/ONE/index.html)

## Running example code

**ROI-based decoding**: 


The script  
`prior_localization/run_scripts/prior_encoding_parallel.py`  
provides an example entry point for running the prior-encoding decoding pipeline in parallel, restricted to user-defined brain region(s) of interest.

To use this script:

1. Select region(s) of interest
   ```python
   ROI_LIST = ["ACAd"]  # set to the ROI(s) you want to analyze
2. Specify session IDs
   eids = [
   
        # MOp example sessions
        #
        # "36280321-555b-446d-9b7d-c2e17991e090",
        # "4aa1d525-5c7d-4c50-a147-ec53a9014812",
        # "5455a21c-1be7-4cae-ae8e-8853a8d5f55e",
        # "81a78eac-9d36-4f90-a73a-7eb3ad7f770b",
        # "9e9c6fc0-4769-4d83-9ea4-b59a1230510e",
        # "bd456d8f-d36e-434a-8051-ff3997253802",
        # "cf43dbb1-6992-40ec-a5f9-e8e838d0f643",
        
        # ACAd example sessions
        "78b4fff5-c5ec-44d9-b5f9-d59493063f00",
        "a4000c2f-fa75-4b3e-8f06-a7cf599b87ad",
    ]

**Plotting corrected R² for a regio**: 

An example plotting script is provided in  
`corrected_R2_plot_single_region.py`.

This script loads the aggregated ROI summary file produced by
`prior_encoding_parallel.py` and visualizes corrected R2
(session R2 minus pseudoR2) across RT groups for a selected region.

