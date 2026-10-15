# Native Clarabel solver (ABI 2)

This Rust `cdylib` links Clarabel 0.11.1 and exposes one generic entry point,
`clarabel_conic_solve`, for problems of the form

    minimise 1/2 x'Px + q'x   subject to   Ax + s = b,  s in {0}^z x R+^l x SOC(d1) x ... x SOC(dk)

The planners in `lander2d/planner.py` and `lander3d/planner.py` assemble the powered-descent SOCP in Python
and hand the CSC arrays (through `common/conic.py`) to this DLL, so the native and Python paths solve
identical matrices. Inputs are validated (sorted CSC, upper-triangular P,
finite data, cone sizes summing to the row count) before Clarabel sees them.

Build the release DLL on Windows:

```powershell
.\native\build_solver.ps1
```

The script writes `clarabel_hover.dll` to the project root (the historic file
name and `clarabel_hover_*` identity symbols are kept). An ABI-1 DLL from the
previous version is detected and ignored; the controller then uses the Python
`clarabel` package with the same matrices.
