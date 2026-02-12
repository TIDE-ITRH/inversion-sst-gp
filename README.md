# inversion-sst-gp

<img align="right" src="3_observing_system_simulation_experiment/outputs/osse_instance_fully_observed.png" alt="drawing" width="400"/>


Physics-informed Gaussian process inversion of sea surface temperature to predict submesoscale near-surface ocean currents, based on the paper by Rick de Kreij et al.

We present a novel method to estimate fine-scale ocean surface currents using satellite sea surface temperature (SST) data. Our approach uses Gaussian process (GP) regression guided by the tracer transport equation, providing not only current predictions but also associated uncertainty. The model effectively handles noisy and incomplete SST data (e.g., due to cloud cover).

**This repository is a fork of the main package.** This fork is intended for use and contains functions to fit individual scenes and time series using `xarray` objects.

---

## Instructions

You need to sign up for NASA Earthdata to download Himawar data using this package (https://urs.earthdata.nasa.gov/)

To reproduce the results, please follow these steps:

1. **Clone this repository** and navigate to the project root (active users can fork the repository).

2. **Install package:**  
    - Recommended: set up a virtual environment using [Poetry](https://python-poetry.org/docs/) and run  
      ```bash
      poetry install
      ``` 
    - Alternative: Conda + PiP 
      ```bash
      conda create -n sstinv python=3.12
      conda activate sstinv
      ```
      ```bash
      pip install ./pkg
      ```
      or install an editable version
      ```bash
      pip install -e ./pkg
      ```  
      Plus for Jupyter install ipykernel.     

3. **Prepare the data:** Download the Himawari-9 data (see notebooks in examples).

4. **Run the project sequentially:** Run the GP regression fit and predict functions (also in example notebooks).


## Abstract

Direct in situ measurements of sea surface currents (SSC) at submesoscales (1-100 km) are challenging. For this reason, one often employs inversion techniques to infer SSC from temperature data, which are straightforward to obtain. However, existing inversion methods have a limited consideration of the underlying physical processes, and do not adequately account for uncertainty. Here, we present a physics-based statistical inversion model to predict submesoscale SSC using remotely sensed sea surface temperature (SST) data. Our approach employs Gaussian process (GP) regression that is informed by a two-dimensional tracer transport equation. Our method yields a predictive distribution of SSC, from which we can generate an ensemble of SSC to construct both predictions and prediction intervals. Our approach incorporates prior knowledge of the SSC length scales and variances elicited from a numerical model; these are subsequently refined using the SST data. The framework naturally handles noisy and spatially irregular SST data (e.g., due to cloud cover), without the need for pre-filtering.  We validate the inversion model through an observing system simulation experiment, which demonstrates that GP-based statistical inversion outperforms existing methods, especially when the measurement signal-to-noise ratio is low.  When applied to Himawari-9 satellite SST data over the eastern Indian Ocean, our method successfully resolves SSC down to the sub-mesoscale. We anticipate our framework will be used to improve understanding of fine-scale ocean dynamics, and to facilitate the coherent propagation of uncertainty into downstream applications such as ocean particle tracking.
