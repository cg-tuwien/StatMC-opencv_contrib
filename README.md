# A Statistical Approach to Monte Carlo Denoising: OpenCV Denoiser

This repository is a copy of [OpenCV's contrib repository](https://github.com/opencv/opencv_contrib) that includes our implementation of the CUDA denoiser for our research paper ["A Statistical Approach to Monte Carlo Denoising" [Sakai et al. 2024]](https://www.cg.tuwien.ac.at/StatMC).
We use this denoiser in our [rendering implementation](https://github.com/cg-tuwien/StatMC).
It is implemented in the [`cudaimgproc` module](modules/cudaimgproc) (in [`modules/cudaimgproc/src/stat_denoiser.cpp`](modules/cudaimgproc/src/stat_denoiser.cpp) and [`modules/cudaimgproc/src/cuda/stat_denoiser.cu`](modules/cudaimgproc/src/cuda/stat_denoiser.cu)).

With the focus on research, this code is not intended for production.
We appreciate your feedback, questions, and reports of any issues you encounter; feel free to [contact us](https://www.cg.tuwien.ac.at/staff/HiroyukiSakai)!


## Build Instructions

### Prerequisites

We developed our denoiser using [CUDA 12.3](https://developer.nvidia.com/cuda-12-3-0-download-archive) and [OpenCV 4.8.1](https://github.com/opencv/opencv/releases/tag/4.8.1).
Note that [later CUDA versions (>= 12.4) are incompatible with OpenCV 4.8.1](https://github.com/opencv/opencv_contrib/issues/3690).
We highly recommend using OpenCV 4.8.1 to match the version of this repository.

For reproducing the results presented in our paper, we recommend using Clang 16.0.6 on Ubuntu 22.04 LTS or Linux Mint 20 (as used for the paper).
While we have successfully tested GCC 11.4.0, it produces slightly different results.

### Building OpenCV

1.  Navigate to your OpenCV build directory.

2.  Create the CMake buildsystem:

    ```bash
    cmake \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_STANDARD=17 \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS} -march=native" \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -march=native" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++ \
    -DCMAKE_CUDA_ARCHITECTURES=<cuda_architecture_number> \
    -DWITH_CUDA=ON \
    -DWITH_CUBLAS=ON \
    -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
    -DBUILD_LIST=cudaarithm,cudev,cudaimgproc,highgui,ximgproc \
    <opencv_source_directory>
    ```
    Here, `<cuda_architecture_number>` must match the architecture number of your graphics card.
    For example, for RTX 3080 Ti and A40 GPUs, the number is 86.
    `<opencv_contrib>` and `<opencv_source_directory>` must point to the directories containing the `opencv_contrib` and `opencv` repositories.
    Note that we only build the modules required for our denoiser.
    With the CMake flag `-DMEMFNC=1` it is possible to enable [Moon et al.'s confidence-interval approach [Moon et al. 2013]](https://doi.org/10.1111/cgf.12004) instead of ours.
    Note that using this flag globally switches to their approach, thereby completely disabling our denoiser.
    To reproduce the results reported in our paper for their approach, it is necessary to change the significance level to 0.002 (from 0.005); this is done by assigning `&t_002_quantiles[0]` to `*t_quantiles` (and uncommenting the corresponding line right above) [here](modules/cudaimgproc/src/cuda/stat_denoiser.cu#L67)
    Furthermore, Box-Cox transformation must be disabled by setting [this flag](https://github.com/HiroyukiSakai/StatMC/blob/master/src/statistics/statpath.cpp#L1043) to `false`.

3.  Build:
    ```bash
    make -j 16
    ```


## Usage

To learn how to use our denoiser, see the simple example provided under [`samples/stat_denoiser/`](samples/stat_denoiser).
Our denoiser is able to efficiently denoise multiple images at once with a single kernel call, which requires some buffer management and preparation, as can be seen in our example.

### Building the Example

1.  Create a build directory somewhere:
    ```bash
    mkdir stat_denoiser
    cd stat_denoiser/
    ```

2.  Create the CMake buildsystem:
    ```bash
    cmake <opencv_contrib>/samples/stat_denoiser/
    ```
    `<opencv_contrib>` must point to the directory containing the `opencv_contrib` repository.
    You may have to provide the build directory if OpenCV with our denoiser has not been installed on your system:
    ```bash
    cmake -DOpenCV_DIR=<opencv_build> <opencv_contrib>/samples/stat_denoiser/
    ```
    Here, `<opencv_build>` must be specified absolutely.

3.  Download the sample images for testing the denoiser:
    ```bash
    ./_download-example-images.sh
    ```

4.  Build:
    ```bash
    make -j 16
    ```

5.  Run the denoiser:
    ```bash
    ./stat_denoiser_example
    ```

Denoised images are output to `staircase-0-16-film-f.pfm` and `staircase-1-16-film-f`.


## Acknowledgments

We thank Lukas Lipp for fruitful discussions, Károly Zsolnai-Fehér and Jaroslav Křivánek for valuable contributions to early versions of this work, and Bernhard Kerbl for help with our CUDA implementation.
This work has received funding from the Vienna Science and Technology Fund (WWTF) project ICT22-028 ("Toward Optimal Path Guiding for Photorealistic Rendering") and the Austrian Science Fund (FWF) project F 77 (SFB "Advanced Computational Design").
