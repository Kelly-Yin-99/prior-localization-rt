
## Prior Localization – RT Subgroup Extension

This repository is an **extension** of the original  
[prior-localization](https://github.com/int-brain-lab/prior-localization)  
repository by Findling, Hubert et al.

## Original paper  
Findling, Hubert et al. (2023), *Brain-wide representations of prior information in mouse decision-making*

This fork extends the original decoding pipeline to perform **reaction-time (RT)–based subgroup analyses**
(e.g. fast / normal / slow trials) **while preserving the original decoding framework and model assumptions**.

---

## Conceptual overview

The core goal of this extension is to ask:

> **Does neural encoding of prior information differ across reaction-time subgroups?**



## RT subgroup decoding pipeline


1. **Fit a single decoder per session using all trials**
   - All trials are used to estimate decoding weights  
   - This yields more stable parameters

2. **Decode prior / target for every trial**
   - Produces decoded values for every trials, regardless of RT subgroup

3. **Generate pseudo-sessions**
   - Create `N = 200` pseudo-sessions by shuffling trial labels 
   - Obtainx null distributions of decoded values

4. **RT subgroup analysis happens *after decoding***
   - Trials are split into `fast`, `normal`, and `slow`
   - The *same subgroup mask* is applied to:
     - real decoded trials
     - pseudo-session decoded trials

5. **For each subgroup and session**
   - Compute:
     - Pearson correlation between decoded prior vs target (Real, pseudo, corrected and Z-score normalized)
     - Real, pseudo and corrected \(R^2\)

---

## What is different in this fork?

Compared to the original implementation, this repository adds:

- **Reaction-time–based trial stratification**
  - Trials are grouped into `fast`, `normal`, and `slow` using configurable RT thresholds

- **RT computation from wheel kinematics**
  - Reaction time is derived from wheel-velocity–based movement onset detection  
  - Implemented in  
    `prior_localization/prior_localization/my_rt.py`

- **Post-hoc subgroup decoding analysis**

- **Pseudo-session–based null distributions**
  - Enables principled statistical comparison of subgroup effects

- **Parallel processing across sessions**

All original decoding logic, estimators, and assumptions remain unchanged unless explicitly stated.

---

## Metrics reported

For each **session × ROI × RT subgroup**, the pipeline computes:

### Correlation-based metrics

- **Real Pearson correlation**
  \[
  r_{\text{real}}
  \]

- **Pseudo-session distribution**
  \[
  r_{\text{pseudo}}^{(1..200)}
  \]

- **z-scored Pearson correlation**
  \[
  z = \frac{r_{\text{real}} - \mu(r_{\text{pseudo}})}{\sigma(r_{\text{pseudo}})}
  \]

- **Corrected Pearson correlation**
  \[
  r_{\text{corr}} = r_{\text{real}} - \mu(r_{\text{pseudo}})
  \]

### Variance-based metrics

- **Real \(R^2\)**
- **Pseudo-session \(R^2\) distribution**
- **Corrected \(R^2\)**
  \[
  R^2_{\text{corr}} = R^2_{\text{real}} - \mu(R^2_{\text{pseudo}})
  \]

These metrics quantify **how strongly real neural encoding exceeds chance expectations**.

---

## Regions of interest

This extension focuses on regions where Findling et al. reported strong prior encoding:

- MOp  
- MOs  
- ACAd  
- ORBvl  

Only sessions satisfying the following criteria are analyzed:

- ≥ 401 total trials
- ≥ 10 trials in each RT subgroup
- Good spike-quality clusters
- ≥ 10 units in the ROI

---

## Dependencies

The code has been tested on:

- Ubuntu 20.04 / 22.04  
- Rocky Linux 8.8  
- macOS 13+

Using Python 3.8–3.10.

Required packages are listed in  
[requirements.txt](https://github.com/int-brain-lab/prior-localization/blob/main/requirements.txt).

---


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


To run RT subgroup decoding for a **single session**, use:
`prior_localization/prior_localization/run_scripts/prior_encoding_subgroup_single_session.py`  


To use this script:

1. Select region(s) of interest
   ```python
   ROI_LIST = ["MOp"]  # set to the ROI(s) you want to analyze, can be multiple e.g. ["MOp", "ACAd"]
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

To inspect the output for this single-session run, use:

`prior_localization/prior_localization/run_scripts/check_prior_subgroup.py`  

To run RT subgroup decoding for **all sessions** in MOp, use:

`prior_localization/prior_localization/run_scripts/prior_encoding_subgroups_parallel.py`

To use this script:
1. Select region(s) of interest
   ```python
   ROI_LIST = ["MOp", "MOs", "ACAd", "ORBvl"]
2. The sessions ids are also provided in this file and is ready to run
   ```python
   DEFAULT_EID_DIR = SCRIPT_DIR / "prior_localization_sessionfit_output" / "roi_eids_all"
   DEFAULT_EID_TXT = DEFAULT_EID_DIR / "eids_union.txt"
   DEFAULT_EID_PKL = DEFAULT_EID_DIR / "eids_union.pkl"
   
To visualize the distribution of subgroup decoding metrics (e.g. Pearson correlation), use:

`prior_localization/prior_localization/run_scripts/plot_pearsonR_and_Rsquare.py`



