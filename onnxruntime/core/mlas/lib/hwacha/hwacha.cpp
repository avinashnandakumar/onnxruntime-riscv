#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdexcept>

#include "util.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
void HwachaDepthWiseConv(const size_t batch_size,
                         const size_t group_count,
                         const size_t channels,
                         const size_t in_height, const size_t in_width,
                         const size_t filter_count,
                         const size_t kernel_height, const size_t kernel_width,
                         const size_t pad_left_height, const size_t pad_left_width,
                         const size_t pad_right_height, const size_t pad_right_width,
                         const size_t dilation_height, const size_t dilation_width,
                         const size_t stride_height, const size_t stride_width,
                         const size_t out_height, const size_t out_width,
                         const int8_t* input, const int8_t* filter, const int32_t* bias,
                         int8_t* output, const float real_multiplier) {
  
  // printf("Starting Hwacha Depth Wise Convolution!\n");

  printf("Padding Left Height: %li Padding Left Widdth: %li Padding Right Height: %li Padding Right Width: %li\n",
         pad_left_height, pad_left_width, pad_right_height, pad_right_width);

  int16_t temp_output[channels];
  for (size_t k = 0; k < channels; k++) {
      temp_output[k] = (int16_t)0;
  }

  int16_t debug_output[out_height * out_width * channels];
  //set zero in hwacha
  //for (size_t m = 0; m < out_height; m++) {
  //  for (size_t k = 0; k < out_width * channels; k++) {
      //output[m * channels * out_width + k] = (int16_t)0;
      //debug_output[m * channels * out_width + k] = (int32_t)0;
  //  }
  //}

  setvcfg(0, 1, 4, 1);
  int consumed = setvlen(channels);
  printf("Consumed length: %i\n", consumed);
  printf("Real Multiplier: %f\n", real_multiplier);

  int input_idx = 0;
  int input_idy = 0;

  int8_t* input_ptr = (int8_t*)input;

  for (int output_idy = 0; output_idy < out_height; output_idy += 1) {
    for (int output_idx = 0; output_idx < out_width * channels; output_idx += channels) {
      for (size_t k = 0; k < channels; k++) {
        temp_output[k] = (int16_t)0;
      }
      asm volatile("vmca va0, %0" : : "r" (temp_output));  //temp_output
    
      input_idy = output_idy - pad_left_height;
      for (int filter_idy = 0; filter_idy < kernel_height; filter_idy++) {
        if (output_idy == 0 && filter_idy == 0 && pad_left_height == 1) {
          printf("pad top buffer zero. output_y: %i output_x: %i filter_y: %i \n",
                 output_idy, output_idx, filter_idy);
          input_idy += 1;
          continue;
        }
        else if(output_idy == out_height-1 && filter_idy == kernel_height-1 && pad_right_height == 1){
          printf("pad bottom buffer zero. output_y: %i output_x: %i filter_y: %i \n", output_idy, output_idx, filter_idy);
          //input_idy += 1;
          continue;
        }

        input_idx = output_idx - pad_left_width * channels;
        for (int filter_idx = 0; filter_idx < kernel_width * channels; filter_idx += channels) {
          if (output_idx == 0 && filter_idx == 0 && pad_left_width == 1) {
            printf("pad left buffer zero. output_y: %i output_x: %i filter_y: %i filter_x: %i \n",
                   output_idy, output_idx, filter_idy, filter_idx);
            input_idx += channels;
            continue;
          }
          else if(output_idx == out_width*channels-channels && filter_idx == kernel_width*channels-channels && pad_right_width == 1){
            printf("pad right buffer zero. output_y: %i output_x: %i filter_y: %i filter_x: %i \n", output_idy, output_idx, filter_idy, filter_idx);
            input_idx += channels;
            continue;
          }

          // printf("Input Values: %i %i; Filter Values: %i %i Input_IDX: %i  Input_IDY: %i \n",
          //        input[input_idx + input_idy * in_width * channels], input[1 + input_idx + input_idy * in_width * channels],
          //        filter[filter_idx + filter_idy * kernel_width * channels], filter[1 + filter_idx + filter_idy * kernel_width * channels],
          //        input_idx, input_idy);

          asm volatile("vmca va1, %0" : : "r" (input_ptr + input_idx + input_idy * in_width * channels));
          asm volatile("vmca va2, %0" : : "r" (filter + filter_idx + filter_idy * kernel_width * channels));
          asm volatile("la t0, vtest3" : : : "t0");
          asm volatile("lw t1, 0(t0)");
          asm volatile("vf 0(t0)");
          
          input_idx += channels;
        }
        input_idy += 1;
      }
      
      asm volatile("vmca va0, %0" : : "r" (temp_output));  //output
      asm volatile("vmca va1, %0" : : "r" (output + output_idx + output_idy * out_width * channels));  //output
      asm volatile("vmca va2, %0" : : "r" (debug_output + output_idx + output_idy * out_width * channels));  //output
      // asm volatile("flw fa0, %0, 0" : : "r" (real_multiplier));
      // asm volatile("fmv.x.s a4, fa0");
      asm volatile("vmcs vs1, %0" : : "r" (real_multiplier));  //real_multiplier
      asm volatile("la t0, vtest6" : : : "t0");
      asm volatile("lw t1, 0(t0)");
      asm volatile("vf 0(t0)");
      
    }
  }
  asm volatile("fence");

  printf("\nFinished Hwacha Depthwise Convolution! \n");
  
  // for (size_t m = 0; m < out_height; m++) {
  //   for (size_t k = 0; k < out_width; k++) {
  //     for (size_t c = 0; c < channels; c++){
  //       if(c == 0){
  //         printf("%i ", debug_output[m * channels * out_width + k*channels + c], debug_output[m * channels * out_width + k*channels + c]);
  //       }
  //     }
  //   }
  //   printf("\n");
  // }
}
#pragma GCC diagnostic pop