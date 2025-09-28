#ifndef LIGHTHOUSE_INIT_C_H
#define LIGHTHOUSE_INIT_C_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Initializes poseidon.
MLIR_CAPI_EXPORTED void lighthouseRegisterDialects(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // LIGHTHOUSE_INIT_C_H
