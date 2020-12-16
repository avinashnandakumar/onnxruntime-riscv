// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv.h"
#include "core/providers/hwacha/hwacha_fwd.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/providers/hwacha/hwacha_execution_provider.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/common/safeint.h"

namespace onnxruntime {
namespace hwacha {

ONNX_OPERATOR_KERNEL_EX(
    Conv,
    kOnnxDomain,
    11,
    kHwachaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1, 10,
    kHwachaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

// ONNX_OPERATOR_TYPED_KERNEL_EX(
//     QLinearConv,
//     kOnnxDomain,
//     10,
//     int8_t,
//     kHwachaExecutionProvider,
//     KernelDefBuilder()
//         .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
//         .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
//         .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())
//         .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
//     QLinearConv<StorageOrder::NCHW>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv_nhwc,
    kOnnxDomain,
    10,
    int8_t,
    kHwachaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv<StorageOrder::NHWC>);

template <typename T>
Status Conv<T>::Compute(OpKernelContext* context) const {
  ORT_UNUSED_PARAMETER(context);
  ORT_ENFORCE(false, "This is a dummy operator");

  return Status::OK();
}

template <>
Status QLinearConv<StorageOrder::NHWC>::Compute(OpKernelContext* context) const {
  printf("CALLED INTO HWACHA PROVIDER\n");
  
  profiling::Profiler& profiler = static_cast<OpKernelContextInternal*>(context)->GetProfiler();
  bool profiling_enabled = profiler.IsEnabled();

  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(3);

  // validate offsets
  auto X_zero_point = context->Input<Tensor>(2);
  auto W_zero_point = context->Input<Tensor>(5);
  auto Y_zero_point = context->Input<Tensor>(7);
  ORT_ENFORCE(IsScalarOr1ElementVector(X_zero_point),
              "QLinearConv : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(W_zero_point),
              "QLinearConv : filter zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(Y_zero_point),
              "QLinearConv : result zero point must be a scalar or 1D tensor of size 1");

  auto X_zero_point_value = *(X_zero_point->template Data<int8_t>());
  auto W_zero_point_value = *(W_zero_point->template Data<int8_t>());
  auto Y_zero_point_value = *(Y_zero_point->template Data<int8_t>());
  ORT_ENFORCE(X_zero_point_value == 0, "Systolic can only handle zero offset for input");
  ORT_ENFORCE(W_zero_point_value == 0, "Systolic can only handle zero offset for filter");
  ORT_ENFORCE(Y_zero_point_value == 0, "Systolic can only handle zero offset for result");

  // validate scale
  auto X_scale = context->Input<Tensor>(1);
  auto W_scale = context->Input<Tensor>(4);
  auto Y_scale = context->Input<Tensor>(6);
  ORT_ENFORCE(IsScalarOr1ElementVector(X_scale),
              "QLinearConv : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(W_scale),
              "QLinearConv : filter scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(Y_scale),
              "QLinearConv : result scale must be a scalar or 1D tensor of size 1");

  auto X_scale_value = *(X_scale->template Data<float>());
  auto W_scale_value = *(W_scale->template Data<float>());
  auto Y_scale_value = *(Y_scale->template Data<float>());

  ORT_ENFORCE(Y_scale_value != 0, "Y_scale_value cannot be 0");
  ORT_ENFORCE(W_scale_value != 0, "W_scale_value cannot be 0");
  ORT_ENFORCE(X_scale_value != 0, "X_scale_value cannot be 0");

  const float real_multiplier = (X_scale_value * W_scale_value) / Y_scale_value;

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor* B = nullptr;
  if (num_inputs == 9) {
    B = context->Input<Tensor>(8);
  }

  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[3];
  const int64_t M = W->Shape()[3];
  // ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShapeNHWC(X, W));
  ORT_ENFORCE(B == nullptr || B->Shape().NumDimensions() == 1, "Bias is not 1D");
  ORT_ENFORCE(B == nullptr || B->Shape().Size() == M, "1D Bias does not match M");

  std::vector<int64_t> kernel_shape;
  TensorShape oihw_w_shape = {W->Shape()[3], W->Shape()[2], W->Shape()[0], W->Shape()[1]};
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(oihw_w_shape, kernel_shape));

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims_nchw({N, M});
  TensorShape input_shape = X->Shape().Slice(1, 3);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims_nchw));
  std::vector<int64_t> Y_dims = {Y_dims_nchw[0], Y_dims_nchw[2], Y_dims_nchw[3], Y_dims_nchw[1]};
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(1, 3);

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const size_t kernel_rank = kernel_shape.size();
  ORT_ENFORCE(kernel_rank == 2, "NHWC cannot handle kernel rank other than 2 atm");

  // If we can run on Systolic, do so!
  // if (TryConvOnSystolic(
  //         static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode(),
  //         dilations,
  //     pads, strides, conv_attrs_.group, X, W, B, Y, fused_relu_, real_multiplier)) {
  //   return Status::OK();
  // }

  fprintf(stderr, "INPUT SHAPE %s\n", input_shape.ToString().c_str());
  fprintf(stderr, "KERNEL SHAPE %s\n", W->Shape().ToString().c_str());
  fprintf(stderr, "OUTPUT SHAPE %s\n", Y->Shape().ToString().c_str());

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C * input_image_size;
  const int64_t Y_offset = (Y->Shape().Size() / Y->Shape()[0]);
  const int64_t B_offset = static_cast<int>(M / conv_attrs_.group);
  const int64_t kernel_dim = C / conv_attrs_.group * kernel_size;
  const int64_t col_buffer_size = C * output_image_size * kernel_size;

  //DEBUG Values
  size_t Nu, He, Wi, Ch, In, Ou, ch1, ch2;
  ch1 = 0;
  ch2 = 1; 
  
  
  // string file_name = (Node().Name()+ "_ref.out").c_str();
  // std::replace(file_name.begin(), file_name.end(), '/', '_');
  

  // FILE *debug_out;
  // debug_out = fopen("debug_verbose_ref/all_layers_student.out","a");
  // //debug_out = fopen(("debug_verbose_student_2/" + file_name).c_str(),"w");

  

  // if(!debug_out) printf("FILE DID NOT OPEN! %s\n", ("debug_verbose/" + file_name).c_str());
  // The col buffer is stored in HWC order as well - the height and width, and
  // kernel_dim.
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  BufferUniquePtr col_buffer;

  if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    auto col_data = alloc->Alloc(SafeInt<size_t>(sizeof(int8_t)) * col_buffer_size);
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));
  } else {
    printf("1x1 case!\n");
  }

  auto* col_buffer_data = static_cast<int8_t*>(col_buffer.get());

  const auto* Xdata = X->template Data<int8_t>();
  const auto* Wdata = W->template Data<int8_t>();
  const auto* Bdata = B != nullptr ? B->template Data<int32_t>() : nullptr;
  auto* Ydata = Y->template MutableData<int8_t>();

  for (int image_id = 0; image_id < N; ++image_id) {
    TimePoint start_time;
    if (profiling_enabled) {
      start_time = profiler.StartTime();
    }
    
    printf("Real Multiplier: %f\n", real_multiplier);
    if (conv_attrs_.group > 1 && conv_attrs_.group == C ){
      printf("Called Hwacha DWC\n");
      HwachaDepthWiseConv(0,//batchsize
          conv_attrs_.group,
          C,
          input_shape[0], input_shape[1],
          0, //filtercount
          kernel_shape[0], kernel_shape[1],
          // 1,1,
          // 1,1,
          pads[0], pads[1],
          pads[2], pads[3],
          dilations[0], dilations[1],
          strides[0], strides[1],
          output_shape[0], output_shape[1],
          Xdata,
          Wdata,
          Bdata,
          Ydata,
          real_multiplier);

          if (profiling_enabled) {
            // std::string channels_string;
            // channel_string = std::to_string(static_cast<int>(M / conv_attrs_.group)) +
            //                   ", " + std::to_string(static_cast<int>(output_image_size)) + ", " +
            //                   std::to_string(static_cast<int>(kernel_dim));
            profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                          Node().Name(),
                                          start_time,
                                          {{"op_name", KernelDef().OpName()},
                                            {"dimensions", std::to_string(static_cast<int>(C))},
                                            {"provider", KernelDef().Provider()}});
            start_time = profiler.StartTime();
        }

      }
      Xdata += X_offset;
      Ydata += Y_offset;
  }
  
  return Status::OK();
}


template <>
Status QLinearConv<StorageOrder::NCHW>::Compute(OpKernelContext* context) const {
  ORT_UNUSED_PARAMETER(context);
  ORT_ENFORCE(false, "This is a dummy operator");

  return Status::OK();
}
}  // namespace hwacha
}  // namespace onnxruntime