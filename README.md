# MLIR Lighthouse Project

This project implements the RFC: https://discourse.llvm.org/t/rfc-mlir-project-lighthouse/86738

_"In essence, this project should guide you through using MLIR for your own projects, showing the way, but not forcing you to follow a particular path. Essentially, the role of a lighthouse."_

### Disclaimer

This project uses LLVM to test and validate various pipelines and assumptions about the MLIR project, but it is not part of any LLVM releases, nor is a dependency for MLIR to operate.

You should use this project to guide you through MLIR pipelines, schedules and transforms, as well as understanding how to connect ingress frameworks and how to execute the egress on appropriate hardware.

## Project Overview

The project is separated into three parts:
* **Ingress:** Frameworks that convert source objects (code, models, designs) into MLIR files that the MLIR project can consume.
* **Scheduler**: The core MLIR component that allows one to choose particular schedules, transforms, dialects to construct pipelines and pass that MLIR ingress through and reach some egress format (LLVM IR, SPIR-V, etc).
* **Runtime**: Environments that can consume the egress format, install/load the appropriate libraries and tools, and execute the output on some target (hardware, simulator, further tools).

The upstream _lighthouse_ project needs to keep the three parts restricted to upstream / publicly available technology. In essence, public ingress and conversion projects, upstream MLIR dialects and transforms, and public execution engines that can be automatically installed and executed without going through private repositories, license agreement, etc.

A downstream fork of the _lighthouse_ project could extend any and all of the three parts to reach private repositories and tools, execute downstream schedules, load private dialects, etc.

The main purposes of this project, in chronological order, are:
1. To **test and validate the existing assumptions in the upstream MLIR repository**, by encoding common ingress paths, transform schedules, pipelines, target differentiation and basic execution.
2. To help MLIR developers **find common patterns on their pipelines**, and propose actions to merge and reuse upstream code for the same purposes.
3. Once common patterns are detected, to **discuss and agree on dialect design, canonical shapes, and common invariants**, to promote upstream and downstream collaboration on the same grounds.
4. Build a solid base to guide downstream projects (open or closed source) to **fork this project and build on top of it**, making it easier to separate upstream/downstream parts and make it easier to upstream the delta to MLIR and/or the _lighthouse_.
5. In time, this could eventually be the **seed for official upstream tooling that uses MLIR in production environments**, like Clang is to LLVM.

### Upstream MLIR / upstream Lighthouse

One key point in the proposal was to not hold _"load bearing"_ code in this repository, but instead, upstream it to MLIR proper and _use_ it here.

It should be fine to have schedule descriptions, aggregation transforms and passes that _use_ the upstream MLIR transforms and passes, but we should _not_ add actual transforms, dialects and passes here to _complement_ the MLIR story.

## Testing Framework

This project should have the same initial purpose as the LLVM Test Suite [https://github.com/llvm/llvm-test-suite], but for MLIR.

## Current Status

Since this is a new project, we're separating the work in branches that will eventually be merged to `main`.
After the initial period, when we have a reasonable testing infrastructure, we can fork & create pull requests.

The planned work is:
* **Ingress**: PyTorch to Linalg extraction on branch [TBD].
* **Harness**: Python harness to bind the MLIR tools, dialects and transforms together on branch [harness](https://github.com/llvm/lighthouse/tree/harness).
* **Schedule**: MLIR schedules to combine into pipelines and combination strategy on branch [TBD].
* **Runtime**: Dependencies, tools and environments to run on various hardware on branch [TBD].
* **Build System**: CMake / Bazel magic to check dependencies, track repositories, run CI on branch [TBD].
