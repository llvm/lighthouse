Importer for the MLIR harness.

Importing MLIR files into the project should be done in a two-stage process:
  1. Build/Install requirements and verify that all dependencies are met.
  2. Run the importer and output the file into the `cache` directory.

These steps are independent to running the harness, to allow different environments to produce the MLIR files than the ones running the compiler.
They can easily be done in one go, on the same machine, by just calling them one after the other.
