# Setup Work

## FHEON

This project is based on the first version of **FHEON** provided on GitHub:

https://github.com/stamcenter/fheon

The build process for this project is identical to that of FHEON, as this work extends the original codebase.  
If anything is unclear, please refer to the official FHEON documentation:

https://fheon.pqcsecure.org/getting_started.html

---

## Build OpenFHE

First, set up **OpenFHE** by following the official installation instructions:

https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html

---

## Create and Build Binaries

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```


## Run Binaries
`./tresnet34N32`
You should change the `tresnet34N32` to any equivalent configuration provided in the paper. 



## Run Binaries
`./tresnet34N32`
You should change the `tresnet34N32` to any equivalent configuration provided in the paper. 
