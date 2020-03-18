// Copyright 2019 MAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "core/OperatorRegister.h"
#include "util/MAIUtil.h"

#ifdef MAI_NEON_ENABLED
#include "neon/GemmNeon.h"
#endif
#include "ref/GemmRef.h"

namespace MAI {
namespace Op {
namespace CPU {

#define GEMM_FUNC_PARAM const float* aPtr, const float* bPtr,\
    const float* cPtr, float* oPtr, int M, int N, int K

#define GEMM_FUNC_DECALRE void(GEMM_FUNC_PARAM)

template<typename T, bool transpose_a, bool transpose_b>
class GemmImpl {
public:
     static void gemm(const std::vector<shape_t>& aShape, const T* a,
            const std::vector<shape_t>& bShape, const T* b,
            const std::vector<shape_t>& cShape, const T* c,
            const std::vector<shape_t>& outputShape, T* output);
};

template<typename T>
class GemmImpl<T, false, false> {
public:
     static void gemm(const std::vector<shape_t>& aShape, const T* a,
            const std::vector<shape_t>& bShape, const T* b,
            const std::vector<shape_t>& cShape, const T* c,
            const std::vector<shape_t>& outputShape, T* output) {
        for(shape_t i = 0; i < outputShape[0]; ++i) {
            for(shape_t j = 0; j < outputShape[1]; ++j) {
                for(shape_t k = 0; k < aShape[1]; ++k) {
                    output[i * outputShape[1] + j] += *(a + (i * aShape[1] + k)) * *(b + k * bShape[1] + j);
                }
                output[i * outputShape[1] + j] += c[j];
            }
        }
    }
};

template<typename T>
class GemmImpl<T, true, false> {
public:
     static void gemm(const std::vector<shape_t>& aShape, const T* a,
            const std::vector<shape_t>& bShape, const T* b,
            const std::vector<shape_t>& cShape, const T* c,
            const std::vector<shape_t>& outputShape, T* output) {
        for(shape_t i = 0; i < outputShape[0]; ++i) {
            for(shape_t j = 0; j < outputShape[1]; ++j) {
                for(shape_t k = 0; k < aShape[0]; ++k) {
                    output[i * outputShape[1] + j] += *(a + (k * aShape[1] + i)) * *(b + k * bShape[1] + j);
                }
                output[i * outputShape[1] + j] += c[j];
            }
        }
    }
};

template<typename T>
class GemmImpl<T, false, true> {
public:
     static void gemm(const std::vector<shape_t>& aShape, const T* a,
            const std::vector<shape_t>& bShape, const T* b,
            const std::vector<shape_t>& cShape, const T* c,
            const std::vector<shape_t>& outputShape, T* output) {
        for(shape_t i = 0; i < outputShape[0]; ++i) {
            for(shape_t j = 0; j < outputShape[1]; ++j) {
                for(shape_t k = 0; k < aShape[1]; ++k) {
                    output[i * outputShape[1] + j] += *(a + (i * aShape[1] + k)) * *(b + j * bShape[1] + k);
                }
                output[i * outputShape[1] + j] += c[j];
            }
        }
    }
};

template<typename T>
class GemmImpl<T, true, true> {
public:
     static void gemm(const std::vector<shape_t>& aShape, const T* a,
            const std::vector<shape_t>& bShape, const T* b,
            const std::vector<shape_t>& cShape, const T* c,
            const std::vector<shape_t>& outputShape, T* output) {
        for(shape_t i = 0; i < outputShape[0]; ++i) {
            for(shape_t j = 0; j < outputShape[1]; ++j) {
                for(shape_t k = 0; k < aShape[0]; ++k) {
                    output[i * outputShape[1] + j] += *(a + (k * aShape[1] + i)) * *(b + j * bShape[1] + k);
                }
                output[i * outputShape[1] + j] += c[j];
            }
        }
    }
};


template<typename T>
class Gemm : public Operator {
public:
    Gemm() : mGemmParam(NULL), mRunFirst(true) {
#ifdef MAI_NEON_ENABLED
        mNoTransANoTransBFunc = NEON::Gemm<T, false, false>::gemm;
#else
        mNoTransANoTransBFunc = Ref::Gemm<T, false, false>::gemm;
#endif
    }

    virtual ~Gemm() {
        if (mGemmParam != NULL) {
            delete mGemmParam;
            mGemmParam = NULL;
        }
    }

    MAI_STATUS init() override {
        return MAI_SUCCESS;
    }

    void setParam(Param* param) override {
        mGemmParam = reinterpret_cast<GemmParam*>(param);
    }

    MAI_STATUS run() override {
        //TODO:(gavinchen) support broadcast
        const Tensor* tensorA = getInputTensor(0);
        const Tensor* tensorB = getInputTensor(1);
        const Tensor* tensorC = getInputTensor(2);
        Tensor* output = getOutputTensor(0);
        MAI_OP_RUN_FIRST_START
        MAI_CHECK_NULL(tensorA);
        MAI_CHECK_NULL(tensorB);
        MAI_CHECK(tensorC != NULL, "Gemm mat c must be exists");// TODO:(gavinchen) c can be NULL
        MAI_CHECK_NULL(output);
        MAI_CHECK_NULL(mGemmParam);
        MAI_CHECK(tensorA->dimSize() == 2, "Gemm mat a must be 2-d");
        MAI_CHECK(tensorB->dimSize() == 2, "Gemm mat b must be 2-d");
        MAI_OP_RUN_FIRST_END
        const T* t1Data = tensorA->data<T>();
        const T* t2Data = tensorB->data<T>();
        const T* t3Data = tensorC->data<T>();

        std::vector<shape_t> outputShape(tensorA->dimSize());
        if (!mGemmParam->transA && !mGemmParam->transB) {
            outputShape[0] = tensorA->dim(0);
            outputShape[1] = tensorB->dim(1);
            output->resize(outputShape);
            output->zero();
            T* outputData = output->mutableData<T>();
            //GemmImpl<T, false,false>::gemm(tensorA->shape(), t1Data, tensorB->shape(), t2Data,
            //        tensorC->shape(), t3Data, outputShape, outputData);
            mNoTransANoTransBFunc(t1Data, t2Data, t3Data, outputData, tensorA->dim(0),
                    tensorB->dim(1), tensorA->dim(1));
        } else if (!mGemmParam->transA && mGemmParam->transB) {
            outputShape[0] = tensorA->dim(0);
            outputShape[1] = tensorB->dim(0);
            output->resize(outputShape);
            output->zero();
            T* outputData = output->mutableData<T>();
            GemmImpl<T, false,true>::gemm(tensorA->shape(), t1Data, tensorB->shape(), t2Data,
                    tensorC->shape(), t3Data, outputShape, outputData);
        } else if (mGemmParam->transA && !mGemmParam->transB) {
            outputShape[0] = tensorA->dim(1);
            outputShape[1] = tensorB->dim(1);
            output->resize(outputShape);
            output->zero();
            T* outputData = output->mutableData<T>();
            GemmImpl<T, true,false>::gemm(tensorA->shape(), t1Data, tensorB->shape(), t2Data,
                    tensorC->shape(), t3Data, outputShape, outputData);
        } else if (mGemmParam->transA && mGemmParam->transB) {
            outputShape[0] = tensorA->dim(1);
            outputShape[1] = tensorB->dim(0);
            output->resize(outputShape);
            output->zero();
            T* outputData = output->mutableData<T>();
            GemmImpl<T, true,true>::gemm(tensorA->shape(), t1Data, tensorB->shape(), t2Data,
                    tensorC->shape(), t3Data, outputShape, outputData);
        }
        return MAI_SUCCESS;
    }
protected:
    std::function<GEMM_FUNC_DECALRE> mNoTransANoTransBFunc;

    std::function<GEMM_FUNC_DECALRE> mNoTransATransBFunc;

private:
    GemmParam* mGemmParam;
    bool mRunFirst;
};

template<typename T>
class GemmBMorden : public Gemm<T> {
public:
    GemmBMorden() : Gemm<T>(){
#ifdef MAI_NEON_ENABLED
        this->mNoTransANoTransBFunc = NEON::Gemm<T, false, false>::gemm_b_morden;
#endif
    }

    ~GemmBMorden(){}

};


template<typename T>
class GemmRef : public Gemm<T> {
public:
    GemmRef() : Gemm<T>(){
        this->mNoTransANoTransBFunc = Ref::Gemm<T, false, false>::gemm;
        ALOGI("GemmRef11 mNoTransANoTransBFunc: == NULL =%d", this->mNoTransANoTransBFunc == NULL);
    }

    ~GemmRef(){}

};

void registerGemm() {
    ALOGI("registerGemm");
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(GEMM).build()), float, Gemm);
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(GEMM).setExtraInfo("ref").build()), float, GemmRef);
    MAI_REGISTER_OP((OpContextBuilder().setOperatorType(GEMM).setExtraInfo("b_morden").build()), float, GemmBMorden);
}

} // namespace CPU
} // namespace Op
} // namespace MAI
