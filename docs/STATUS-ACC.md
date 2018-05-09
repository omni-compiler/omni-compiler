Implementation Status version 1.3.0
---------------------------------------
# General
Current implementaion supports almost parts of OpenACC 1.0 and some feature of OpenACC 2.0.
The implementation supports only CUDA as backend but we are implementing OpenCL version now.

# Not supported features
* worker level parallelization in compute region. (only gang and vector are supported)
* cache directive
* device_resident clause in declare directive

# Supported features of OpenACC 2.0
* atomic directive
* enter_data and exit_data directives

# Notes
* In OpenACC 2.0, async-ids are different for sync kernel and async kernel. However, async-ids are same for sync and async kernel in our compiler.
* Device numbers are 1-based following OpenACC 1.0

# Options
## Environment variable
* OMNI_ACC_NUM_GANGS_LIMIT=N
    * Set limitation of number of gangs. "N=0" means no limitation. "N>0" means limiting number of gangs to be less than or equal to N.

# Known issues
* Compilation fails when data constructs with if clause are nested.
