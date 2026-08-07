from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='FlashKNN',
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["FlashKNN"],
    package_dir={"FlashKNN": "functions"},
    ext_modules=[
        CUDAExtension('FlashKNN.CuFun', [
        'csrc/flash_knn_query_dynamic_load.cu',
        'csrc/flash_knn_query_global_memory.cu',
        'csrc/flash_knn_query_GMPS.cu',
        'csrc/flash_knn_query.cu',
        'csrc/flash_knn_construction.cu',
        'csrc/api.cpp'
        ],
        # Do not hard-code sm_86.  torch's extension builder emits code for
        # TORCH_CUDA_ARCH_LIST when it is set, or for the visible GPU when it
        # is not. This supports Ada/L20 (sm_89), Hopper/H20 (sm_90), and
        # other architectures without editing setup.py.
        extra_compile_args={'cxx': ['-g', "-O3", "-mavx2", "-funroll-loops"],
                            'nvcc': ['-O2', "-Xptxas", "-v", "-lineinfo"]})
    ],
    cmdclass={'build_ext': BuildExtension},
    include_dirs=["csrc/"]
)
