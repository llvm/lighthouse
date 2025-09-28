#include "lighthouse-c/Init.h"
#include "mlir-c/Dialect/Func.h"

MLIR_CAPI_EXPORTED void lighthouseRegisterDialects(MlirDialectRegistry registry) {
  // TODO: Probably have a function call to the C++ lib to register things
  // here. This is just a placeholder for now.
  MlirDialectHandle funcDialect = mlirGetDialectHandle__func__();
  mlirDialectHandleInsertDialect(funcDialect, registry);
}
