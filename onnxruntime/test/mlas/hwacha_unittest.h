#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wformat"
#pragma GCC diagnostic ignored "-Wtype-limits"


#include <mlas.h>
#define ACC_SCALE(x, scale) \
  ({float y = nearbyint((x) * (scale)); y > INT_MAX ? INT_MAX : (y < INT_MIN ? INT_MIN : (acc_t)y); })
#define MVIN_SCALE(x, scale) \
  (scale == 1.0 ? x : ({float y = nearbyint((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (elem_t)y); }))
#define ELEM_T_MAX SCHAR_MAX
#define ELEM_T_MIN SCHAR_MIN
#define FMT "%d "
template <typename elem_t, typename acc_t>
class MlasHwachaDWCTest : public MlasTestBase {
 private:

  inline elem_t saturate(acc_t num, float scale, bool relu) {
    num = ACC_SCALE(num, scale);
    // Clip result
    num = num > ELEM_T_MAX ? ELEM_T_MAX : (num < ELEM_T_MIN ? ELEM_T_MIN : num);
    if (relu) {
      num = num < 0 ? 0 : num;
    }
    return num;
  }

  int tests_total=0; 
  int tests_passed=0;

 protected:
  void
  Test(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t PaddingRightHeight,
      size_t PaddingRightWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth) {
    printf("Testing: Image Dimensions: %i x %i x %i; Kernel Dimensions: %i x %i x %i; Padding %i; Stride %i;\n", InputChannels, InputHeight, InputWidth, FilterCount, KernelHeight, KernelWidth, PaddingLeftHeight, StrideHeight);
    int64_t OutputHeight64 =
        ((int64_t(InputHeight) + int64_t(PaddingLeftHeight) + int64_t(PaddingRightHeight)) -
         (int64_t(DilationHeight) * (int64_t(KernelHeight) - 1) + 1)) /
            int64_t(StrideHeight) +
        1;
    int64_t OutputWidth64 =
        ((int64_t(InputWidth) + int64_t(PaddingLeftWidth) + int64_t(PaddingRightWidth)) -
         (int64_t(DilationWidth) * (int64_t(KernelWidth) - 1) + 1)) /
            int64_t(StrideWidth) +
        1;

    if (OutputHeight64 <= 0 || OutputWidth64 <= 0) {
      return;
    }

    size_t OutputHeight = size_t(OutputHeight64);
    size_t OutputWidth = size_t(OutputWidth64);

    size_t InputSize = InputHeight * InputWidth;
    size_t KernelSize = KernelHeight * KernelWidth;
    size_t OutputSize = OutputHeight * OutputWidth;

    size_t InputElements = BatchCount * GroupCount * InputChannels * InputSize;
    size_t FilterElements = GroupCount * FilterCount * KernelSize;  // Depthwise InputChannels * KernelSize;
    size_t BiasElements = GroupCount * FilterCount;
    size_t OutputElements = BatchCount * GroupCount * FilterCount * OutputSize;

    //const int8_t* A = BufferA.GetBuffer(K * M);

    const int8_t* Input = BufferInput.GetBuffer(InputElements);
    const int8_t* Filter = BufferFilter.GetBuffer(FilterElements);
    const int32_t* Bias = BufferBias.GetBuffer(BiasElements);
    int8_t* Output = BufferOutput.GetBuffer(OutputElements);
    int8_t* OutputReference = BufferOutputReference.GetBuffer(OutputElements);

    HwachaDepthWiseConv(BatchCount,
                        GroupCount,
                        InputChannels,
                        InputHeight, InputWidth,
                        FilterCount,
                        KernelHeight, KernelWidth,
                        PaddingLeftHeight, PaddingLeftWidth,
                        PaddingRightHeight, PaddingRightWidth,
                        DilationHeight, DilationWidth,
                        StrideHeight, StrideWidth,
                        OutputHeight, OutputWidth,
                        Input,
                        Filter,
                        Bias,
                        Output,
                        1);
	conv_dw(BatchCount, 
		       InputHeight, InputChannels, 
		       InputChannels, OutputHeight,
		       StrideHeight, PaddingLeftHeight, KernelHeight, 
		       Input, Filter, NULL, OutputReference, 
		       false, 1);
    
    //  printf("\n");
    //  printf("input\n");
    
    // int Nu, He, Wi, Ch; 

    // Nu = 1; 
    // He = InputHeight;
    // Wi = InputWidth;
    // Ch = InputChannels;

    // for (int n = 0; n < Nu; n++) {
    //   for (int h = 0; h < He; h++) {
    //     for (int w = 0; w < Wi; w++) {
    //       for (int c = 0; c < Ch; c++) {
    //         // if(c == ch1 || c == ch2){
    //           printf("%i, \t", Input[((n*He + h)*Wi + w)*Ch + c]); 
    //         // }
    //       }
    //     }
    //     printf("\n");
    //   }
    //   printf("\n");
    // }


    //  printf("filter\n");
   

    // Nu = 1; 
    // He = KernelHeight;
    // Wi = KernelWidth;
    // Ch = FilterCount;

    // for (int n = 0; n < Nu; n++) {
    //   for (int h = 0; h < He; h++) {
    //     for (int w = 0; w < Wi; w++) {
    //       for (int c = 0; c < Ch; c++) {
    //         // if(c == ch1 || c == ch2){
    //           printf("%i, \t", Filter[((n*He + h)*Wi + w)*Ch + c]); 
    //         // }
    //       }
    //     }
    //     printf("\n");
    //   }
    //   printf("\n");
    // }

     



    // Nu = 1; 
    // He = OutputHeight;
    // Wi = OutputWidth;
    // Ch = FilterCount;
    // printf("\n");
    // printf("Reference Output:\n");
    // for (int n = 0; n < Nu; n++) {
    //   for (int h = 0; h < He; h++) {
    //     for (int w = 0; w < Wi; w++) {
    //       for (int c = 0; c < Ch; c++) {
    //         // if(c == ch1 || c == ch2){
    //           printf("%p:%i \t", &OutputReference[((n*He + h)*Wi + w)*Ch + c], OutputReference[((n*He + h)*Wi + w)*Ch + c]);
    //           //rintf("%p:%i \t",  &OutputReference[k * FilterCount * OutputWidth + l], OutputReference[k * OutputWidth + l]); 
    //         // }
    //       }
    //     }
    //     printf("\n");
    //   }
    //   printf("\n");
    // }

    // Nu = 1; 
    // He = OutputHeight;
    // Wi = OutputWidth;
    // Ch = FilterCount;
    // printf("\n");
    // printf("Actual Output:\n");
    // for (int n = 0; n < Nu; n++) {
    //   for (int h = 0; h < He; h++) {
    //     for (int w = 0; w < Wi; w++) {
    //       for (int c = 0; c < Ch; c++) {
    //         // if(c == ch1 || c == ch2){
    //           printf("%p:%i \t", &Output[((n*He + h)*Wi + w)*Ch + c], Output[((n*He + h)*Wi + w)*Ch + c]);
    //           //rintf("%p:%i \t",  &OutputReference[k * FilterCount * OutputWidth + l], OutputReference[k * OutputWidth + l]); 
    //         // }
    //       }
    //     }
    //     printf("\n");
    //   }
    //   printf("\n");
    // }

  
    tests_total += 1;
    if (memcmp(Output, OutputReference, sizeof(int8_t) * OutputElements) != 0) {
      printf("mismatch: batch=%zd,group=%zd,input(%zd,%zd,%zd),filter=%zd,kernel(%zd,%zd)!!!\n\n",
             BatchCount, GroupCount, InputChannels, InputHeight, InputWidth, FilterCount,
             KernelHeight, KernelWidth);
    } else {
      tests_passed += 1;
      printf("Test PASS! %i/%i\n\n", tests_passed, tests_total);
    }
  }

  virtual void
  MlasConv2D(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t PaddingRightHeight,
      size_t PaddingRightWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth,
      size_t OutputHeight,
      size_t OutputWidth,
      const float* Input,
      const float* Filter,
      const float* Bias,
      float* Output) {
    int64_t InputShape[] = {int64_t(InputHeight), int64_t(InputWidth)};
    int64_t KernelShape[] = {int64_t(KernelHeight), int64_t(KernelWidth)};
    int64_t DilationShape[] = {int64_t(DilationHeight), int64_t(DilationWidth)};
    int64_t Padding[] = {int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth)};
    int64_t StrideShape[] = {int64_t(StrideHeight), int64_t(StrideWidth)};
    int64_t OutputShape[] = {int64_t(OutputHeight), int64_t(OutputWidth)};

    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = MlasIdentityActivation;

    MLAS_CONV_PARAMETERS Parameters;
    size_t WorkingBufferSize;

    MlasConvPrepare(&Parameters,
                    2,
                    BatchCount,
                    GroupCount,
                    InputChannels,
                    InputShape,
                    KernelShape,
                    DilationShape,
                    Padding,
                    StrideShape,
                    OutputShape,
                    FilterCount,
                    &Activation,
                    &WorkingBufferSize,
                    nullptr);

    MlasConv(&Parameters,
             Input,
             Filter,
             Bias,
             BufferWorking.GetBuffer(WorkingBufferSize),
             Output,
             nullptr);
  }

    //void
    //ReferenceConv2D(
    //    size_t BatchCount,
    //    size_t GroupCount,
    //    size_t InputChannels,
    //    size_t InputHeight,
    //    size_t InputWidth,
    //    size_t FilterCount,
    //    size_t KernelHeight,
    //    size_t KernelWidth,
    //    size_t PaddingLeftHeight,
    //    size_t PaddingLeftWidth,
    //    size_t DilationHeight,
    //    size_t DilationWidth,
    //    size_t StrideHeight,
    //    size_t StrideWidth,
    //    size_t OutputHeight,
    //    size_t OutputWidth,
    //    const int8_t* Input,
    //    const int8_t* Filter,
    //    const int32_t* Bias,
    //    int8_t* Output
    //    )
    //{
    //    size_t InputSize = InputHeight * InputWidth;
    //    size_t OutputSize = OutputHeight * OutputWidth;
    //    size_t KernelSize = KernelHeight * KernelWidth;

    //    size_t K = InputChannels * KernelSize;
    //    size_t Im2ColElements = OutputSize * K;

    //    for (size_t b = 0; b < BatchCount; b++) {

    //        const int8_t* filter = Filter;
    //        const int32_t* bias = Bias;

    //        for (size_t g = 0; g < GroupCount; g++) {

    //            //
    //            // Transform the image using IM2COL and invoke the GEMM.
    //            //

    //            int8_t* Im2Col = BufferIm2Col.GetBuffer(Im2ColElements);
    //            int8_t* Im2ColOut = Im2Col;

    //            for (size_t c = 0; c < InputChannels; c++) {

    //                for (size_t ky = 0; ky < KernelHeight; ky++) {

    //                    for (size_t kx = 0; kx < KernelWidth; kx++) {

    //                        for (size_t oh = 0; oh < OutputHeight; oh++) {

    //                            size_t ih = oh * StrideHeight + ky * DilationHeight - PaddingLeftHeight;

    //                            for (size_t ow = 0; ow < OutputWidth; ow++) {

    //                                size_t iw = ow * StrideWidth + kx * DilationWidth - PaddingLeftWidth;

    //                                *Im2ColOut++ = (ih < InputHeight && iw < InputWidth) ?
    //                                    Input[ih * InputWidth + iw] : 0;
    //                            }
    //                        }
    //                    }
    //                }

    //                Input += InputSize;
    //            }

    //    	MlasGemm(CblasNoTrans, CblasNoTrans, FilterCount, OutputSize, K, 1.0f,
    //                filter, K, Im2Col, OutputSize, 0.0f, Output, OutputSize, threadpool);

    //            //
    //            // Apply the bias.
    //            //

    //            for (size_t f = 0; f < FilterCount; f++) {

    //                float biasValue = *bias++;

    //                for (size_t o = 0; o < OutputSize; o++) {
    //                    *Output++ += biasValue;
    //                }
    //            }

    //            filter += FilterCount * InputChannels * KernelSize;
    //        }
    //    }
    //}
  //void
  //ReferenceConv2D(
  //    size_t BatchCount,
  //    size_t GroupCount,
  //    size_t InputChannels,
  //    size_t InputHeight,
  //    size_t InputWidth,
  //    size_t FilterCount,
  //    size_t KernelHeight,
  //    size_t KernelWidth,
  //    size_t PaddingLeftHeight,
  //    size_t PaddingLeftWidth,
  //    size_t DilationHeight,
  //    size_t DilationWidth,
  //    size_t StrideHeight,
  //    size_t StrideWidth,
  //    size_t OutputHeight,
  //    size_t OutputWidth,
  //    const int8_t* Input,
  //    const int8_t* Filter,
  //    const int32_t* Bias,
  //    int8_t* Output) {
  //  size_t InputSize = InputHeight * InputWidth;
  //  size_t OutputSize = OutputHeight * OutputWidth;
  //  size_t KernelSize = KernelHeight * KernelWidth;
  //  size_t GroupSize = FilterCount * InputSize;

  //  size_t K = InputChannels * KernelSize;
  //  size_t Im2ColElements = OutputSize * K;

  //  for (size_t batch = 0; batch < BatchCount; batch++) {
  //    for (size_t group = 0; group < FilterCount; group++) {
  //      for (size_t channel = 0; channel < InputChannels; channel++) {
  //        for (size_t out_row = 0; out_row < OutputHeight; out_row++) {
  //          for (size_t out_col = channel; out_col < OutputWidth * InputChannels; out_col += InputChannels) {
  //            size_t in_row = out_row * StrideHeight - PaddingLeftHeight;

  //            int32_t result = 0;
  //            //if (params->bias) {
  //            //result = Bias[group];
  //            //}

  //            for (size_t kernel_row = 0; kernel_row < KernelHeight; kernel_row++) {
  //              size_t in_col = out_col * StrideWidth - PaddingLeftWidth;

  //              for (size_t kernel_col = channel; kernel_col < KernelWidth * FilterCount; kernel_col += FilterCount) {
  //                if (in_row >= 0 && in_row < InputHeight && in_col >= 0 && in_col < InputWidth * InputChannels) {
  //                  result += Input[group * GroupSize + in_row * InputWidth * InputChannels + in_col] * Filter[group * GroupSize + kernel_row * KernelWidth * FilterCount + kernel_col];
  //                }
  //                //printf("Filter_IDX: %i; Filter_IDY:  %i; Input_IDX: %i;  Input_IDY: %i; Input Value: %i; Filter Value: %i; Result: %i; \n", kernel_col, kernel_row, in_col, in_row, Input[group*GroupSize + in_row*InputWidth*InputChannels + in_col], Filter[group*GroupSize + kernel_row*KernelWidth*FilterCount + kernel_col], result);

  //                in_col += InputChannels;
  //              }

  //              in_row++;
  //            }

  //            /*
  //                          acc_t abs = result >= 0 ? result : -result;
  //                          int divisor = 1 << params->output_scale;
  //                          acc_t shifted = (abs + divisor/2) >> params->output_scale;
  //                          if (result < 0) {
  //                              shifted = -shifted;
  //                          }
  //                          */

  //            if (result < -128) {
  //              result = -128;
  //            }

  //            //int32_t shifted = ROUNDING_RIGHT_SHIFT(result, params->output_scale);

  //            if (result > 127) {
  //              result = 127;
  //            }
  //            //printf("Output_IDX: %i; Output_IDY: %i; Result Value: %i\n", out_col, out_row, result);
  //            Output[group * GroupSize + out_row * OutputWidth * InputChannels + out_col] = result;
  //            //printf("\n");
  //          }
  //        }
  //      }
  //    }
  //  }
  //}


  void conv_dw(
    int batch_size, int in_dim, int channels, 
    int out_channels, int out_dim, 
    int stride, int padding, int kernel_dim,
    const elem_t* input,
    const elem_t* weight,
    const acc_t * bias,
    elem_t* output,
    bool relu, float output_scale)
{
    bool no_bias = bias == NULL;
    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * stride - padding;

                    acc_t result = 0;
                    if (no_bias==false) {
                        result = bias[channel];
                    }

                    for (int kernel_row = 0; kernel_row < kernel_dim; kernel_row++) {
                        int in_col = out_col * stride - padding;

                        for (int kernel_col = 0; kernel_col < kernel_dim; kernel_col++) {
                            if (in_row >= 0 && in_row < in_dim && in_col >= 0 && in_col < in_dim) {
                                //batch might be wrong
                                //printf("Batch %i; Channel %i; In Row: %i; In Col: %i; Kernel Row: %i; Kernel Col: %i;\n", batch, channel, in_row, in_col, kernel_row, kernel_col);
                                result += input[channels*in_dim*in_row + in_col*channels + channel] * weight[channels*kernel_dim*kernel_row + kernel_col*channels + channel];
                                
                                //printf("Multiplying weight and ipixel %d %d \n", weight[channels*kernel_dim*kernel_row + kernel_col*channels + channel], input[channels*in_dim*in_row + in_col*channels + channel]);
                                //result += input[batch][channel][in_row][in_col] * weight[channel][kernel_row][kernel_col];
                            }

                            in_col++;
                        }

                        in_row++;
                    }
                    //printf("\n");

                    /*
                    acc_t abs = result >= 0 ? result : -result;
                    int divisor = 1 << params->output_scale;
                    acc_t shifted = (abs + divisor/2) >> params->output_scale;
                    if (result < 0) {
                        shifted = -shifted;
                    }
                    */
                    output[batch*out_dim*out_dim*channel + channels*out_dim*out_row + out_col*channels + channel] = saturate(result, output_scale, relu);
                    //output[batch][channel][out_row][out_col] = saturate(result, output_scale, relu);
                    //*(output + (b * out_dim * out_dim + orow * out_dim + ocol) * out_channels + och) =
              
                    // if (result < 0) {
                    //     result = 0;
                    // }
                    
                    // acc_t shifted = ROUNDING_RIGHT_SHIFT(result, params->output_scale);

                    // if (shifted > elem_t_max) {
                    //     shifted = elem_t_max;
                    // }                    

                    // output[batch][channel][out_row][out_col] = shifted;
                }
            }
        }
    }
}

  void reference_conv(
      int batch_size, int in_dim, int in_channels,
      int out_channels, int out_dim,
      int stride, int padding, int kernel_dim,
      const elem_t* input,
      const elem_t* weights,
      const acc_t* bias,
      elem_t* output,
      bool relu, float output_scale) {
    bool no_bias = bias == NULL;

    for (int b = 0; b < batch_size; b++) {
      for (int orow = 0; orow < out_dim; orow++) {
        for (int ocol = 0; ocol < out_dim; ocol++) {
          //printf("New output value\n");
          for (int och = 0; och < out_channels; och++) {
            acc_t opixel = no_bias ? 0 : bias[och];
            //printf("New output channel\n");
            for (int krow = 0; krow < kernel_dim; krow++) {
              const int irow = orow * stride + krow - padding;

              for (int kcol = 0; kcol < kernel_dim; kcol++) {
                const int icol = ocol * stride + kcol - padding;

                for (int kch = 0; kch < in_channels; kch++) {
                  elem_t ipixel = irow < 0 || irow >= in_dim || icol < 0 || icol >= in_dim ? 0 : *(input + (b * in_dim * in_dim + irow * in_dim + icol) * in_channels + kch);

                  elem_t weight = *(weights + (krow * kernel_dim * in_channels + kcol * in_channels + kch) * out_channels + och);
                  printf("Multiplying weight and ipixel %d %d\n", weight, ipixel);
                  opixel += weight * ipixel;
                }
              }
            }
          // printf("Value before saturate %d\n", opixel);
            *(output + (b * out_dim * out_dim + orow * out_dim + ocol) * out_channels + och) =
                saturate(opixel, output_scale, relu);
          }
        }
      }
    }
  }

  MatrixGuardBuffer<int8_t> BufferInput;
  MatrixGuardBuffer<int8_t> BufferFilter;
  MatrixGuardBuffer<int32_t> BufferBias;
  MatrixGuardBuffer<int8_t> BufferOutput;
  MatrixGuardBuffer<int8_t> BufferOutputReference;
  MatrixGuardBuffer<float> BufferWorking;
  MatrixGuardBuffer<int8_t> BufferIm2Col;

 public:
  void
  ExecuteLong(
      void) override {
    // N.B. InputChannels must be a multiple of 4 if the count is greater
    // than the block size.
    // static const unsigned cis[] = { 32, 20, 5, 1 };
    // static const unsigned cos[] = { 64, 15, 1 };
    // static const unsigned is[] = { 27, 11, 5, 1 };

    // Depthwise convolutions.
    printf("Avi's Depthwise Tests\n");
    
    // Parameters
    // Batch Count, Group Count
    // Input Channels, Input Height, Input Width
    // Filter Count, Kernel Height, Kernel Width
    // Padding Left Height, Padding Left Width, Padding Right Height, Padding Right Width
    // Dilation Height, Dilation Width (NOT USED IN KERNEL) 
    // Stride Height, Stride Width
   
    //Input Channels 1; Filter Count 1;
    Test(1, 1, 
	 1, 5, 5, 
	 1, 2, 2, 
	 0,0,0,0,
	 1, 1, 
	 1, 1);  

  //   //Input Channels 1; Filter Count 1;
//    Test(1, 1, 
//         1, 5, 5, 
//	 1, 3, 3, 
//	 0, 0, 0, 0, 
//	 1, 1,
//	 1, 1);  
//    
//    
//    Test(1, 1, 
//   	 1, 5, 5, 
//	 1, 4, 4, 
//	 0, 0, 0, 0, 
//	 1, 1, 
//	 1, 1);  
//
//    Test(1, 1, 
//        5, 224, 224, 
//      5, 8, 8, 
//      0, 0, 0, 0, 
//      1, 1, 
//      1, 1);  
//    
//    //Input Channels 2; Filter Count 2;
//    Test(1, 1, 
//         2, 6, 6, 
//	 2, 2, 2, 
//	 1,1,1,1,
//	 0, 0, 
//	 1, 1); 
//    
//  //   //Input Channels 3; Filter Count 3;
//    Test(1, 1, 
//	 3, 5, 5, 
//	 3, 2, 2, 
//	 0, 0, 0, 0, 
//	 1, 1, 
//	 1, 1); 
//    
//  //   // Conv 2_1 Layer;
//    Test(1, 1, 
//	 32, 112, 112, 
//	 32, 3, 3, 
//	 1, 1, 1, 1, 
//	 1, 1, 
//	 1, 1); 
//    
//  //   // Conv 2_2 Layer;
//    Test(1, 1, 
//	 96, 112, 112, 
//	 96, 3, 3, 
//	 1, 1, 1, 1, 
//	 1, 1, 
//	 2, 2); 
//
//  //   // Conv 3_1 Layer
//    Test(1, 1, 
//         144, 56, 56,
//	 144, 3, 3, 
//	 1, 1, 1, 1, 
//	 1, 1, 
//	 1, 1);

    //for (unsigned i = 16; i < 256; i <<= 1) {

    //Test(1, i, 1, 28, 28, 1, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);

    // Test(1, i, 1, 28, 28, 1, 3, 3, 0, 0, 0, 0, 1, 1, 2, 2);
    // Test(1, i, 1, 28, 28, 1, 3, 3, 0, 0, 0, 0, 2, 2, 1, 1);
    // Test(1, i, 1, 28, 28, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
    // Test(1, i, 1, 28, 28, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    // Test(1, i, 1, 28, 28, 1, i, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    // Test(12, i, 1, 11, 11, 1, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
    //}

    // Test varying FilterCounts.
    // for (unsigned i = 1; i < 128; i++) {
    //     Test(1, 1, 3, 34, 34, i, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
    //     Test(1, 1, 16, 34, 34, i, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
    //     Test(1, 1, 16, 34, 34, i, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    // }

    // for (unsigned i = 1; i <= 32; i++) {
    //     Test(4, 18, 1, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
    //     Test(4, 18, 1, 32, 89, 48, i, 89, 1, 1, 1, 1, 1, 1, 1, 1);
    //     Test(4, 18, 2, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
    // }

    // for (unsigned b = 1; b < 64; b++) {
    //     Test(b, 1, 64, 11, 11, 128, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    // }

    // for (unsigned ic = 0; ic < _countof(cis); ic++) {
    //     for (unsigned ih = 0; ih < _countof(is); ih++) {
    //         for (unsigned iw = 0; iw < _countof(is); iw++) {
    //             fprintf(stderr, "Handling %ux%ux%u\n", cis[ic], is[ih], is[iw]);
    //             for (unsigned fc = 0; fc < _countof(cos); fc++) {
    //                 for (unsigned kh = 1; kh <= 5; kh++) {
    //                     if (kh == 4) continue;
    //                     for (unsigned kw = 1; kw <= 5; kw++) {
    //                         if (kw == 4) continue;
    //                         for (unsigned p0 = 0; p0 <= 3; p0++) {
    //                             for (unsigned p1 = 0; p1 <= 3; p1++) {
    //                                 for (unsigned p2 = 0; p2 <= 3; p2++) {
    //                                     for (unsigned p3 = 0; p3 <= 3; p3++) {
    //                                         for (unsigned dh = 1; dh <= 2; dh++) {
    //                                             for (unsigned dw = 1; dw <= 2; dw++) {
    //                                                 for (unsigned sh = 1; sh <= 2; sh++) {
    //                                                     for (unsigned sw = 1; sw <= 2; sw++) {
    //                                                         Test(1, 1, cis[ic], is[ih], is[iw], cos[fc], kh, kw, p0, p1, p2, p3, dh, dw, sh, sw);
    //                                                     }
    //                                                 }
    //                                             }
    //                                         }
    //                                     }
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
  }

  // public:
  //     void ExecuteShort(void) override
  //     {
  //       size_t K = 4;
  //       size_t M = 4;
  //       //int N = 4;

  //       printf("avi's test\n");
  //       printf("Testing...\n");
  //       const int8_t* A = BufferA.GetBuffer(K * M);
  //       //const int8_t* B = BufferB.GetBuffer(N * K);
  //       //const int32_t* Bias = BufferBias.GetBuffer(N * M);
  //       //int8_t* C = BufferC.GetBuffer(N * M);
  //       //int8_t* CReference = BufferCReference.GetBuffer(N * M);

  //       printf("A matrix:\n");
  //       for (size_t m = 0; m < M; m++) {
  //           for (size_t k = 0; k < K; k++) {
  //               printf("%d ", A[m * K + k]);
  //           }
  //           printf("\n");
  //       }

  //         // Should match precisely for exact multiples of systolic size
  //         // printf("Testing exact dimensions with no divisor\n");
  //         // Test(16, 16, 16, 1, 0);
  //         // Test(1*16, 2*16, 3*16, 1, 0);
  //         // Test(16, 16, 16, 1, 0, /*relu= */ true);
  //         // Test(1*16, 2*16, 3*16, 1, 0, /*relu= */ true);
  //         //
  //         // // Should match preicsely for exact multiples with divisor (right shift)
  //         // printf("Testing exact dimensions with divisor\n");
  //         // Test(16, 16, 16, 4, 0);
  //         // Test(1*16, 2*16, 3*16, 4, 0);
  //         // Test(16, 16, 16, 4, 0, /*relu= */ true);
  //         // Test(1*16, 2*16, 3*16, 4, 0, /*relu= */ true);
  //         //
  //         // printf("Testing non-exact dimensions with divisor\n");
  //         // Test(3, 5, 7, 2, 0);
  //         // Test(89, 79, 83, 4, 0);
  //         // Test(18, 45, 337, 8, 0, /*relu= */ true);
  //         // Test(1697, 2029, 1319, 16, 0, /*relu= */ true);

  //         //HwachaDepthWiseConv(24);
  //     }

  //     void ExecuteLong(void) override
  //     {
  //     }

  //     //MlasHwachaDWCTest();
};
#pragma GCC diagnostic pop
