// © 2024-2025 Hiroyuki Sakai

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)
#error "Could not find CUDA."
#else /* !defined (HAVE_CUDA) || defined (CUDA_DISABLER) */

namespace cv { namespace cuda { namespace device { namespace imgproc {
    namespace stat_denoiser {
        void setup();
        void synchronize(cudaStream_t stream);
        template <typename T>
        void calculate_mean_vars(
            const unsigned short ptrCount,
            const unsigned short width,
            const unsigned short height,
            const PtrStepSzb &nPtrs,
            const PtrStepSzb &m2Ptrs,
            PtrStepSzb meanVarPtrs,
            cudaStream_t stream
        );
        template <typename T>
        void filter(
            const unsigned short ptrCount,
            const unsigned short width,
            const unsigned short height,
            const float dSFactor,
            const unsigned char radius,
            const bool denoiseFilm,
            const PtrStepSzb &nPtrs,
            const PtrStepSzb &meanPtrs,
            const PtrStepSzb &m2Ptrs,
            const PtrStepSzb &m3Ptrs,
            const PtrStepSzb &filmPtrs,
            const PtrStepSzb &film,
            const PtrStepSzb &gBufPtrs,
            const PtrStepSzb &gBufChannelCounts,
            const PtrStepSzb &gBufDRFactors,
            const unsigned char nGBufs,
            PtrStepSzb meanCorrPtrs,
            PtrStepSzb discriminatorPtrs,
            PtrStepSzb filmFilteredPtrs,
            PtrStepSzb filmFiltered,
            cudaStream_t stream
        );
    }
}}}}

void cv::cuda::stat_denoiser::setup() {
    cv::cuda::device::imgproc::stat_denoiser::setup();
}

void cv::cuda::stat_denoiser::synchronize(Stream &stream) {
    cv::cuda::device::imgproc::stat_denoiser::synchronize(StreamAccessor::getStream(stream));
}

template <typename T>
void cv::cuda::stat_denoiser::calculateMeanVars(
    const unsigned short ptrCount,
    const unsigned short width,
    const unsigned short height,
    const PtrStepSzb &nPtr,
    const PtrStepSzb &m2Ptr,
    PtrStepSzb meanVarPtr,
    Stream &stream
) {
    cv::cuda::device::imgproc::stat_denoiser::calculate_mean_vars<T>(
        ptrCount,
        width,
        height,
        nPtr,
        m2Ptr,
        meanVarPtr,
        StreamAccessor::getStream(stream)
    );
}
template
CV_EXPORTS_W void cv::cuda::stat_denoiser::calculateMeanVars<float>(
    const unsigned short ptrCount, const unsigned short width, const unsigned short height, const PtrStepSzb &nPtr, const PtrStepSzb &m2Ptr, PtrStepSzb meanVarPtr, Stream &stream
);
template
CV_EXPORTS_W void cv::cuda::stat_denoiser::calculateMeanVars<float3>(
    const unsigned short ptrCount, const unsigned short width, const unsigned short height, const PtrStepSzb &nPtr, const PtrStepSzb &m2Ptr, PtrStepSzb meanVarPtr, Stream &stream
);

template <typename T>
void cv::cuda::stat_denoiser::filter(
    const unsigned short ptrCount,
    const unsigned short width,
    const unsigned short height,
    const float dSFactor,
    const unsigned char radius,
    const bool denoiseFilm,
    const PtrStepSzb &nPtrs,
    const PtrStepSzb &meanPtrs,
    const PtrStepSzb &m2Ptrs,
    const PtrStepSzb &m3Ptrs,
    const PtrStepSzb &filmPtrs,
    const PtrStepSzb &film,
    const PtrStepSzb &gBufferPtrs,
    const PtrStepSzb &gBufferChannelCounts,
    const PtrStepSzb &gBufferDRFactors,
    const unsigned char nGBufs,
    PtrStepSzb meanCorrPtrs,
    PtrStepSzb discriminatorPtrs,
    PtrStepSzb filmFilteredPtrs,
    PtrStepSzb filmFiltered,
    Stream &stream
) {
    cv::cuda::device::imgproc::stat_denoiser::filter<T>(
        ptrCount,
        width,
        height,
        dSFactor,
        radius,
        denoiseFilm,
        nPtrs,
        meanPtrs,
        m2Ptrs,
        m3Ptrs,
        filmPtrs,
        film,
        gBufferPtrs,
        gBufferChannelCounts,
        gBufferDRFactors,
        nGBufs,
        meanCorrPtrs,
        discriminatorPtrs,
        filmFilteredPtrs,
        filmFiltered,
        StreamAccessor::getStream(stream)
    );
}
template
CV_EXPORTS_W void cv::cuda::stat_denoiser::filter<float>(
    const unsigned short, const unsigned short, const unsigned short, const float, const unsigned char, const bool, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const unsigned char, PtrStepSzb, PtrStepSzb, PtrStepSzb, PtrStepSzb, Stream &
);
template
CV_EXPORTS_W void cv::cuda::stat_denoiser::filter<float3>(
    const unsigned short, const unsigned short, const unsigned short, const float, const unsigned char, const bool, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const unsigned char, PtrStepSzb, PtrStepSzb, PtrStepSzb, PtrStepSzb, Stream &
);

template <typename T>
void cv::cuda::stat_denoiser::filter(
    const unsigned short ptrCount,
    const unsigned short width,
    const unsigned short height,
    const float dSFactor,
    const unsigned char radius,
    const PtrStepSzb &nPtrs,
    const PtrStepSzb &meanPtrs,
    const PtrStepSzb &m2Ptrs,
    const PtrStepSzb &m3Ptrs,
    const PtrStepSzb &filmPtrs,
    const PtrStepSzb &gBufferPtrs,
    const PtrStepSzb &gBufferChannelCounts,
    const PtrStepSzb &gBufferDRFactors,
    const unsigned char nGBufs,
    PtrStepSzb meanCorrPtrs,
    PtrStepSzb discriminatorPtrs,
    PtrStepSzb filmFilteredPtrs,
    Stream &stream
) {
    GpuMat nullMat; // Hack
    cv::cuda::device::imgproc::stat_denoiser::filter<T>(
        ptrCount,
        width,
        height,
        dSFactor,
        radius,
        false,
        nPtrs,
        meanPtrs,
        m2Ptrs,
        m3Ptrs,
        filmPtrs,
        nullMat,
        gBufferPtrs,
        gBufferChannelCounts,
        gBufferDRFactors,
        nGBufs,
        meanCorrPtrs,
        discriminatorPtrs,
        filmFilteredPtrs,
        nullMat,
        StreamAccessor::getStream(stream)
    );
}
template
CV_EXPORTS_W void cv::cuda::stat_denoiser::filter<float>(
    const unsigned short, const unsigned short, const unsigned short, const float, const unsigned char, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const unsigned char, PtrStepSzb, PtrStepSzb, PtrStepSzb, Stream &
);
template
CV_EXPORTS_W void cv::cuda::stat_denoiser::filter<float3>(
    const unsigned short, const unsigned short, const unsigned short, const float, const unsigned char, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const PtrStepSzb &, const unsigned char, PtrStepSzb, PtrStepSzb, PtrStepSzb, Stream &
);

#endif /* !defined (HAVE_CUDA) || defined (CUDA_DISABLER) */
