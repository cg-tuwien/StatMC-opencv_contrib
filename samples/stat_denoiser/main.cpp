// © 2024-2025 Hiroyuki Sakai
//
// Demo for our statistical denoiser, which denoises two renderings with a single kernel call.
// We provide more details here: https://github.com/cg-tuwien/StatMC-opencv_contrib#usage

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using std::string, std::vector;
using cv::Mat, cv::Mat_, cv::imread, cv::IMREAD_UNCHANGED;
using cv::cuda::GpuMat, cv::cuda::PtrStepSzb, cv::cuda::Stream;
using Vec3 = cv::Vec<float, 3>;

struct float3 {
    float x, y, z;
};


void inline alloc(Mat mat, vector<Mat> &mats, vector<GpuMat> &gpuMats) {
    mats.emplace_back(mat);
    gpuMats.emplace_back(GpuMat(mat.rows, mat.cols, mat.type()));
}

void inline alloc(string filename, vector<Mat> &mats, vector<GpuMat> &gpuMats, int type = -1) {
    Mat mat = imread(filename, IMREAD_UNCHANGED);
    mat.convertTo(mat, type);
    alloc(mat, mats, gpuMats);
}

void inline uploadGPUPtrs(vector<GpuMat> &gpuMats, GpuMat &gpuPtrs, Stream stream) {
    Mat gpuPtrsCPU = Mat(1, gpuMats.size(), CV_8UC(sizeof(PtrStepSzb)));
    PtrStepSzb *gpuPtrsCPUPtr = gpuPtrsCPU.ptr<PtrStepSzb>();
    for (auto &gpuMat : gpuMats)
        *gpuPtrsCPUPtr++ = gpuMat;
    gpuPtrs.upload(gpuPtrsCPU, stream);
}

void inline uploadGBufferChannelCounts(vector<GpuMat> &gpuMats, GpuMat &channelCounts, Stream stream) {
    Mat channelCountsCPU = Mat(1, gpuMats.size(), CV_8UC1);
    unsigned char *channelCountsCPUPtr = channelCountsCPU.ptr<unsigned char>();
    for (auto &gpuMat : gpuMats)
        *channelCountsCPUPtr++ = gpuMat.channels();
    channelCounts.upload(channelCountsCPU, stream);
}


int main(int argc, char** argv )
{
    string prefix = "staircase";
    string suffix = "16";
    vector<string> indices = {"0", "1"}; // We denoise two different renderings with these indices.
    int nRenderings = indices.size();

    // Set denoising parameters
    float ciZValue = 1.95996f;
    float sd = 10.f;
    int radius = 20;
    float normalSD = 0.1f;
    float albedoSD = 0.02f;


    // CUDA stream
    Stream stream;

    // Load images and allocate buffers
    vector<Mat> films;
    vector<Mat> ns;
    vector<Mat> means;
    vector<Mat> m2s;
    vector<Mat> m3s;
    vector<Mat> gBuffers;
    vector<Mat> meanCorrs;
    vector<Mat> discriminators;
    vector<Mat> denoisedFilms;

    vector<GpuMat> gpuFilms;
    vector<GpuMat> gpuNs;
    vector<GpuMat> gpuMeans;
    vector<GpuMat> gpuM2s;
    vector<GpuMat> gpuM3s;
    vector<GpuMat> gpuGBuffers;
    vector<GpuMat> gpuMeanCorrs;
    vector<GpuMat> gpuDiscriminators;
    vector<GpuMat> gpuDenoisedFilms;

    for (auto &index : indices) {
        alloc(prefix + "-" + index + "-" + suffix + "-film.pfm",       films, gpuFilms);
        alloc(prefix + "-" + index + "-" + suffix + "-t0-b0-n.pfm",    ns,    gpuNs, CV_32SC1);
        alloc(prefix + "-" + index + "-" + suffix + "-t0-b0-mean.pfm", means, gpuMeans);
        alloc(prefix + "-" + index + "-" + suffix + "-t0-b0-m2.pfm",   m2s,   gpuM2s);
        alloc(prefix + "-" + index + "-" + suffix + "-t0-b0-m3.pfm",   m3s,   gpuM3s);
    }

    // We use the same set of G-buffers to denoise both renderings.
    alloc(prefix + "-0-" + suffix + "-t1-b0-film-mean.pfm", gBuffers, gpuGBuffers);
    alloc(prefix + "-0-" + suffix + "-t2-b0-film-mean.pfm", gBuffers, gpuGBuffers);

    int width = films[0].cols;
    int height = films[1].rows;

    for (auto &index : indices) {
        alloc(Mat_<Vec3>(height, width), meanCorrs,      gpuMeanCorrs);
        alloc(Mat_<Vec3>(height, width), discriminators, gpuDiscriminators);
        alloc(Mat_<Vec3>(height, width), denoisedFilms,  gpuDenoisedFilms);
    }


    // Upload vectors containing pointers to GPU buffers
    GpuMat filmGPUPtrs;
    GpuMat nGPUPtrs;
    GpuMat meanGPUPtrs;
    GpuMat m2GPUPtrs;
    GpuMat m3GPUPtrs;

    GpuMat gBufferGPUPtrs;
    GpuMat gBufferChannelCounts;
    GpuMat gBufferDRFactors;

    GpuMat meanCorrGPUPtrs;
    GpuMat discriminatorGPUPtrs;
    GpuMat denoisedFilmGPUPtrs;

    uploadGPUPtrs(gpuFilms, filmGPUPtrs, stream);
    uploadGPUPtrs(gpuNs,    nGPUPtrs,    stream);
    uploadGPUPtrs(gpuMeans, meanGPUPtrs, stream);
    uploadGPUPtrs(gpuM2s,   m2GPUPtrs,   stream);
    uploadGPUPtrs(gpuM3s,   m3GPUPtrs,   stream);

    uploadGPUPtrs(gpuGBuffers, gBufferGPUPtrs, stream);
    // For the G-buffers, we additionally need to upload the channel counts for the individual buffers.
    uploadGBufferChannelCounts(gpuGBuffers, gBufferChannelCounts, stream);

    uploadGPUPtrs(gpuMeanCorrs,      meanCorrGPUPtrs,      stream);
    uploadGPUPtrs(gpuDiscriminators, discriminatorGPUPtrs, stream);
    uploadGPUPtrs(gpuDenoisedFilms,  denoisedFilmGPUPtrs,  stream);


    // Upload images
    for (int i = 0; i < nRenderings; i++) {
        gpuFilms[i].upload(films[i], stream);
        gpuNs[i]   .upload(ns[i],    stream);
        gpuMeans[i].upload(means[i], stream);
        gpuM2s[i]  .upload(m2s[i],   stream);
        gpuM3s[i]  .upload(m3s[i],   stream);
    }

    for (int i = 0; i < gBuffers.size(); i++)
        gpuGBuffers[i].upload(gBuffers[i], stream);


    // Prepare denoising parameters
    float dsFactor = -.5f / (sd * sd);
    vector drFactors{
        -.5f / (normalSD * normalSD),
        -.5f / (albedoSD * albedoSD)
    };
    gBufferDRFactors.upload(Mat(drFactors), stream);


    // Denoise
    cv::cuda::stat_denoiser::filter<float3>(
        nRenderings, // Number of renderings to denoise
        width, // Width of the renderings
        height, // Height of the renderings
        dsFactor, // Spatial distance factor calculated from spatial filter standard deviation
        radius, // Filter radius
        nGPUPtrs, // Pointers to GPU buffers for sample sizes (n)
        meanGPUPtrs, // Pointers to GPU buffers for means
        m2GPUPtrs, // Pointers to GPU buffers for sums of squared deviations (m2)
        m3GPUPtrs, // Pointers to GPU buffers for sums of cubed deviations (m3)
        filmGPUPtrs, // Pointers to GPU buffers for noisy input images
        gBufferGPUPtrs, // Pointers to GPU buffers for G-buffers
        gBufferChannelCounts, // G-buffer channel counts
        gBufferDRFactors, // Range distance factors calculated from range standard deviations for G-buffers
        gBuffers.size(), // Number of G-buffers
        meanCorrGPUPtrs, // Pointers to GPU buffers for Johnson-corrected means
        discriminatorGPUPtrs, // Pointers to GPU buffers for discriminators
        denoisedFilmGPUPtrs, // Pointers to GPU buffers for denoised output images
        stream // CUDA stream
    );


    // Write denoised images to disk
    for (int i = 0; i < nRenderings; i++) {
        gpuDenoisedFilms[i].download(denoisedFilms[i], stream);
        imwrite(prefix + "-" + indices[i] + "-" + suffix + "-film-f.pfm", denoisedFilms[i]);
    }

    return 0;
}
