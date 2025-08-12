Schedules for `mlir-opt` and friends.

There are three types of schedules:
  * `lib`: A library of sub-schedules that do one particular thing with a set of parameters.
  * `group`: A schedule that calls the sub-schedules in the library with the appropriate parameters from target descriptors.
  * `pipeline`: A super-schedule that calls group schedules in order to build a whole pipeline.

Each level can be tested independently (IR and execution).
There can be multiple pipelines calling the grouping or library schedules with different parameters.