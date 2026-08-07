from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys
from pathlib import Path
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

setup(
    name='PyTorchCudaFlann',
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["PyTorchCudaFlann"],
    package_dir={"PyTorchCudaFlann": "CudaFlann"},
    ext_modules=[
        CUDAExtension('PyTorchCudaFlann.CuFun', [
        'src/cpp/torch_api.cpp',
        'src/cpp/flann/algorithms/kdtree_cuda_3d_index.cu',
        # 'src/cpp/flann/flann.cpp',
        # 'src/cpp/flann/flann_cpp.cpp'
        ],
        libraries=['lz4'],
        library_dirs=[str(Path(sys.prefix) / "lib")],
        extra_compile_args={'cxx': ["-O3", "-mavx2", "-funroll-loops"],
                            # Let TORCH_CUDA_ARCH_LIST select the local GPU
                            # (8.6 on RTX 3090, 8.9 on L20, 9.0 on H20).
                            'nvcc': ['-O2']})
    ],
    include_dirs = [str(Path(__file__).resolve().parent / "src" / "cpp"),
                    str(Path(sys.prefix) / "include")],
    cmdclass={'build_ext': BuildExtension},
)
