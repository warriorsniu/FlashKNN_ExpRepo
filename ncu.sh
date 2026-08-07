export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LIBRARY_PATH"  

ncu \
--metrics smsp__sass_average_branch_targets_threads_uniform.pct,\
smsp__thread_inst_executed_per_inst_executed.ratio \
-k "regex:(?i).*knn.*" \
--print-summary per-kernel \
python test_query.py

ncu \
--metrics dram__sectors_read.sum,\
dram__sectors_write.sum \
-k "regex:(?i).*knn.*" \
--print-summary per-kernel \
python test_query.py



# KnnKernel
# Grid_Knn_Query_dynamic_load_kernel
# sm__sass_branch_targets_threads_divergent.avg