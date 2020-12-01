// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hwacha/hwacha_execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "hwacha_fwd.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

namespace hwacha {

//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHwachaExecutionProvider, kOnnxDomain, 10, int8_t, QLinearConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHwachaExecutionProvider, kOnnxDomain, 10, int8_t, QLinearConv_nhwc);
// Forward declarations of op kernels
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHwachaExecutionProvider, kOnnxDomain, 1, 10, Conv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHwachaExecutionProvider, kOnnxDomain, 11, Conv);

static Status RegisterHwachaKernels(KernelRegistry& kernel_registry) {
    static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHwachaExecutionProvider, kOnnxDomain, 1, 10, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHwachaExecutionProvider, kOnnxDomain, 11, Conv)>,
      //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHwachaExecutionProvider, kOnnxDomain, 10, int8_t, QLinearConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHwachaExecutionProvider, kOnnxDomain, 10, int8_t, QLinearConv_nhwc)>,
    };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }
  printf("Registered HWACHA Kernels\n");
  return Status::OK();
}

struct KernelRegistryAndStatus {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  Status st;
};

KernelRegistryAndStatus GetHwachaKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = RegisterHwachaKernels(*ret.kernel_registry);
  return ret;
}

}  // namespace hwacha

char HwachaExecutionProvider::GetAcceleratorMode() const {
  return provider_info_.accelerator_mode;
}

void HwachaExecutionProvider::InsertFusedRules(FuseRuleFn rule) {
  fuse_rules_.push_back(rule);
}

std::shared_ptr<KernelRegistry> HwachaExecutionProvider::GetKernelRegistry() const {
  static hwacha::KernelRegistryAndStatus k = onnxruntime::hwacha::GetHwachaKernelRegistry();
  //throw if the registry failed to initialize
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}

std::unique_ptr<IDataTransfer> HwachaExecutionProvider::GetDataTransfer() const {
  return onnxruntime::make_unique<CPUDataTransfer>();
}

std::vector<std::unique_ptr<ComputeCapability>>
HwachaExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                    const std::vector<const KernelRegistry*>& kernel_registries) const {
  
  printf("Checking capability HWACHA\n");

  std::vector<std::unique_ptr<ComputeCapability>> result; 
      // result = IExecutionProvider::GetCapability(graph, kernel_registries);


  for (auto& node : graph.Nodes()) {
    for (auto registry : kernel_registries) {
      if ((node.OpType() == "QLinearConv") || KernelRegistry::HasImplementationOf(*registry, node, Type())) {
        const auto* group_attr = graph_utils::GetNodeAttribute(node, "group");
        int group_count = group_attr->i();
        if(node.InputDefs()[0]->Shape() != nullptr && node.InputDefs()[0]->Shape()->dim_size() == 4){ //Call GraphUtils::NodeArgIsConstant
          if(group_count <= 1 || group_count != node.InputDefs()[0]->Shape()->dim(1).dim_value()){
            continue;
          }
          else
          {
            printf("Found implementation optype: %s name: %s \n", node.OpType().c_str(), node.Name().c_str());
            std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
            sub_graph->nodes.push_back(node.Index());
            result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
          }
        }

        
        
        break;
      }
    }
  }

  for (auto& rule : fuse_rules_) {
    rule(graph, result);
  }
  return result;
}

void HwachaExecutionProvider::SetupFusedRules() {}

}  // namespace onnxruntime
