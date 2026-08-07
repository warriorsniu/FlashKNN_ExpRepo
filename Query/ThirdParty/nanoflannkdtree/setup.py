from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

setup(
    name='PyTorchNanoFlann',
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["PyTorchNanoFlann"],
    package_dir={"PyTorchNanoFlann": "TorchFlann"},
    ext_modules=[
        CppExtension('PyTorchNanoFlann.Kdtree', [
        'csrcs/cutils.cpp',
        'csrcs/kdtree.cpp'
        ],
        extra_compile_args={'cxx': ["-O3", "-mavx2", "-funroll-loops"]})
    ],
    cmdclass={'build_ext': BuildExtension},
)
