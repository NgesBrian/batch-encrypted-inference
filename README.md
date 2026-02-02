

# Setup WORK

# FHEON
We started this project by cloning the first version of FHEON provided on github at [https://github.com/stamcenter/fheon](https://github.com/stamcenter/fheon)
The build process of this project is exactly the same as that of FHEON as we extended the project. 
If anything is unclear in this project checkout the [FHEON documentation](https://fheon.pqcsecure.org/getting_started.html)


# Build OpenFHE
First Setup OpenFHE following the instructions provided at [Building OpenFHE](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html)


## Create and build binaries
`
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
`

## Run Binaries
`./tresnet34N32`
You should change the `tresnet34N32` to any equivalent configuration provided in the paper. 
