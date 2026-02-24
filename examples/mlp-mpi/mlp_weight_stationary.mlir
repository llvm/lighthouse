module attributes {{mpi.dlti = #dlti.map<"MPI:Implementation" = "MPICH", "MPI:comm_world_size" = {P}, "MPI:comm_world_rank" = {R}> }} {{
  shard.grid @grid0(shape = {grid}) {{sym_visibility = "private"}}
  func.func @{func_name}(%arg0: tensor<{M}x{K}xf32>, %arg1: tensor<{K}x{N}xf32>, %arg2: tensor<{N}x{K}xf32>) -> tensor<{M}x{K}xf32> attributes {{llvm.emit_c_interface}} {{
    %cst = arith.constant 0.000000e+00 : f32
    
    %sharding_arg0 = shard.sharding @grid0 split_axes = {split_act} : !shard.sharding
    %sharding_arg1 = shard.sharding @grid0 split_axes = {split_win} : !shard.sharding
    %sharding_arg2 = shard.sharding @grid0 split_axes = {split_wout} : !shard.sharding

    %sharding_mm0a_mm1c = shard.sharding @grid0 split_axes = {split_mm0a_mm1c} : !shard.sharding
    %sharding_mm0_c = shard.sharding @grid0 split_axes = {split_mm0_c} : !shard.sharding

    %sharding_sigmoid = shard.sharding @grid0 split_axes = {split_sigmoid} : !shard.sharding

    %sharding_r = shard.sharding @grid0 split_axes = {split_r} : !shard.sharding

    %sharded = shard.shard %arg0 to %sharding_arg0 : tensor<{M}x{K}xf32>
    %sharded_6 = shard.shard %arg1 to %sharding_arg1 : tensor<{K}x{N}xf32>
    %sharded_7 = shard.shard %arg2 to %sharding_arg2 : tensor<{N}x{K}xf32>
  
    %0 = tensor.empty() : tensor<{M}x{N}xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>

    %sharded_11 = shard.shard %sharded to %sharding_mm0a_mm1c annotate_for_users : tensor<{M}x{K}xf32>
    %sharded_13 = shard.shard %1 to %sharding_mm0_c annotate_for_users : tensor<{M}x{N}xf32>
    %2 = linalg.matmul ins(%sharded_11, %sharded_6 : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>) outs(%sharded_13 : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>

    %sharded_15 = shard.shard %2 to %sharding_sigmoid annotate_for_users : tensor<{M}x{N}xf32>
    %3 = tosa.sigmoid %sharded_15 : (tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>

    %4 = tensor.empty() : tensor<{M}x{K}xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<{M}x{K}xf32>) -> tensor<{M}x{K}xf32>

    %sharded_22 = shard.shard %5 to %sharding_mm0a_mm1c annotate_for_users : tensor<{M}x{K}xf32>
    %6 = linalg.matmul ins(%3, %sharded_7 : tensor<{M}x{N}xf32>, tensor<{N}x{K}xf32>) outs(%sharded_22 : tensor<{M}x{K}xf32>) -> tensor<{M}x{K}xf32>

    %sharded_24 = shard.shard %6 to %sharding_arg0 annotate_for_users : tensor<{M}x{K}xf32>
    return %sharded_24 : tensor<{M}x{K}xf32>
  }}

  func.func @alloc_act() -> (tensor<{M}x{K}xf32>) attributes {{llvm.emit_c_interface}} {{
    %a = tensor.empty() : tensor<{M}x{K}xf32>
    %sharding_act = shard.sharding @grid0 split_axes = {split_act} : !shard.sharding
    %sharded_act = shard.shard %a to %sharding_act : tensor<{M}x{K}xf32>
    %ret_a = shard.shard %sharded_act to %sharding_act annotate_for_users : tensor<{M}x{K}xf32>
    return %ret_a : tensor<{M}x{K}xf32>
  }}

  func.func @alloc_win() -> (tensor<{K}x{N}xf32>) attributes {{llvm.emit_c_interface}} {{
    %b = tensor.empty() : tensor<{K}x{N}xf32>
    %sharding_win = shard.sharding @grid0 split_axes = {split_win} : !shard.sharding
    %sharded_win = shard.shard %b to %sharding_win : tensor<{K}x{N}xf32>
    %ret_win = shard.shard %sharded_win to %sharding_win annotate_for_users : tensor<{K}x{N}xf32>
    return %ret_win : tensor<{K}x{N}xf32>
  }}

  func.func @alloc_wout() -> (tensor<{N}x{K}xf32>) attributes {{llvm.emit_c_interface}} {{
    %c = tensor.empty() : tensor<{N}x{K}xf32>
    %sharding_wout = shard.sharding @grid0 split_axes = {split_wout} : !shard.sharding
    %sharded_wout = shard.shard %c to %sharding_wout : tensor<{N}x{K}xf32>
    %ret_wout = shard.shard %sharded_wout to %sharding_wout annotate_for_users : tensor<{N}x{K}xf32>
    return %ret_wout : tensor<{N}x{K}xf32>
  }}

  func.func @alloc_r() -> (tensor<{M}x{K}xf32>) attributes {{llvm.emit_c_interface}} {{
    %a = tensor.empty() : tensor<{M}x{K}xf32>
    %sharding_r = shard.sharding @grid0 split_axes = {split_r} : !shard.sharding
    %sharded_r = shard.shard %a to %sharding_r : tensor<{M}x{K}xf32>
    %ret_a = shard.shard %sharded_r to %sharding_r annotate_for_users : tensor<{M}x{K}xf32>
    return %ret_a : tensor<{M}x{K}xf32>
  }}

  func.func @dealloc_2d(%arg0: memref<?x?xf32>) attributes {{llvm.emit_c_interface}} {{
    memref.dealloc %arg0 : memref<?x?xf32>
    return
  }}

  func.func @gather_act(%arg0: tensor<{M}x{K}xf32>) -> tensor<{M}x{K}xf32> attributes {{llvm.emit_c_interface}} {{
    %sharding = shard.sharding @grid0 split_axes = {split_act} : !shard.sharding
    %sharding_g = shard.sharding @grid0 split_axes = [[]] : !shard.sharding
    %sharded = shard.shard %arg0 to %sharding : tensor<{M}x{K}xf32>
    %sharded_g = shard.shard %sharded to %sharding_g annotate_for_users : tensor<{M}x{K}xf32>
    return %sharded_g : tensor<{M}x{K}xf32>
  }}

  func.func @gather_win(%arg0: tensor<{K}x{N}xf32>) -> tensor<{K}x{N}xf32> attributes {{llvm.emit_c_interface}} {{
    %sharding = shard.sharding @grid0 split_axes = {split_win} : !shard.sharding
    %sharding_g = shard.sharding @grid0 split_axes = [[]] : !shard.sharding
    %sharded = shard.shard %arg0 to %sharding : tensor<{K}x{N}xf32>
    %sharded_g = shard.shard %sharded to %sharding_g annotate_for_users : tensor<{K}x{N}xf32>
    return %sharded_g : tensor<{K}x{N}xf32>
  }}

  func.func @gather_wout(%arg0: tensor<{N}x{K}xf32>) -> tensor<{N}x{K}xf32> attributes {{llvm.emit_c_interface}} {{
    %sharding = shard.sharding @grid0 split_axes = {split_wout} : !shard.sharding
    %sharding_g = shard.sharding @grid0 split_axes = [[]] : !shard.sharding
    %sharded = shard.shard %arg0 to %sharding : tensor<{N}x{K}xf32>
    %sharded_g = shard.shard %sharded to %sharding_g annotate_for_users : tensor<{N}x{K}xf32>
    return %sharded_g : tensor<{N}x{K}xf32>
  }}
}}
