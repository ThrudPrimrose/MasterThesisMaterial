cur_dir=$(pwd)
ln -sf $cur_dir/flow.py $cur_dir/gemmforge/flow.py
ln -sf $cur_dir/benchmark_dense_sparse_cuda.py $cur_dir/gemmforge/benchmark_dense_sparse_cuda.py
ln -sf $cur_dir/benchmark_sparse_dense_cuda.py $cur_dir/gemmforge/benchmark_sparse_dense_cuda.py
ln -sf $cur_dir/parse.py $cur_dir/gemmforge/parse.py

