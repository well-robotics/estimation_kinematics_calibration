# MATLAB Prototype

This folder contains a MATLAB prototype of the bi-level calibration pipeline for a **2-D planar five-link walker**. This MATLAB version is closer to the original development workflow. It is mainly intended for:

- reading the algorithm in a smaller prototype,
- visualizing estimator trajectories and iteration-wise behavior,
- debugging modeling choices and intermediate quantities.

## Files

- **`main.m`**  
  Runs the planar walker simulation, injects noise, and launches the outer calibration loop.

- **`estimation_FIE.m`**  
  Defines the full-information estimator (FIE) used as the inner problem.

- **`plot_FIE.m`**  
  Plots estimated trajectories against ground truth for quick inspection.

## Relation to the repository

This MATLAB folder is not the main maintained implementation. It is best viewed as a compact reference prototype.

This prototype is closely related to the MATLAB workflow used in the **STRIDE** planar biped codebase.

For the broader planar-walking MATLAB environment, including generated expressions, simulator utilities, and controller components, please refer to:

`https://github.com/well-robotics/STRIDE/tree/main/Software/Matlab`

Conceptually:

- **STRIDE MATLAB** provides the broader planar robot software stack and control policy.
- **This folder** provides a smaller Full Information Estimator / Moving Horizon Estimator + calibration-oriented prototype.

## Note

This folder is **not fully standalone**.

The main script expects additional MATLAB code and external dependencies to already be available on your MATLAB path. In particular, `main.m` uses paths such as:

- `Expression`
- `library`
- `controller`

So if you are setting this up from scratch, you will likely need to use STRIDE, or your own equivalent local MATLAB setup, and update the `addpath(...)` lines accordingly.

## Dependencies

You will typically need:

- MATLAB
- CasADi for MATLAB
- IPOPT
- qpOASES
- YALMIP
- MOSEK
- STRIDE

Depending on your local setup, you may also need generated expressions and other planar walker utilities.

Before running, edit `main.m` and update any machine-specific paths.

The script will then simulate the walker, build noisy measurements, solve the inner FIE/MHE, run outer-loop calibration iterations, and generate plots or exported results.

## This version is good for

This MATLAB prototype is especially useful for:

- quick visualization,
- comparing curves across iterations,
- inspecting intermediate quantities,
- finite-difference or sanity-check style debugging.

If you encounter issues or would like a shorter minimal example, please feel free to contact:**denglinc@umich.edu**
