import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

def make_cpu_ext(name, module, sources, extra_args=[]):
    return CppExtension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        extra_compile_args={"cxx": [] + extra_args},
    )

if __name__ == "__main__":
    setup(
        name="mmdet3d",
        packages=find_packages(),
        include_package_data=True,
        package_data={"mmdet3d.ops": ["*/*.so"]},
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        license="Apache License 2.0",
        ext_modules=[
            make_cpu_ext(
                name="sparse_conv_ext",
                module="mmdet3d.ops.spconv",
                extra_args=["-w", "-std=c++14"],
                sources=[
                    "src/all.cc",
                    "src/reordering.cc",
                    "src/indice.cc",
                    "src/maxpool.cc",
                ],
            ),
            make_cpu_ext(
                name="bev_pool_ext",
                module="mmdet3d.ops.bev_pool",
                sources=[
                    "src/bev_pool.cpp",
                ],
            ),
            make_cpu_ext(
                name="iou3d_cpu",
                module="mmdet3d.ops.iou3d",
                sources=[
                    "src/iou3d.cpp",
                ],
            ),
            make_cpu_ext(
                name="voxel_layer",
                module="mmdet3d.ops.voxel",
                sources=[
                    "src/voxelization.cpp",
                    "src/scatter_points_cpu.cpp",
                    "src/voxelization_cpu.cpp",
                ],
            ),
            make_cpu_ext(
                name="roiaware_pool3d_ext",
                module="mmdet3d.ops.roiaware_pool3d",
                sources=[
                    "src/roiaware_pool3d.cpp",
                    "src/points_in_boxes_cpu.cpp",
                ],
            ),
            make_cpu_ext(
                name="ball_query_ext",
                module="mmdet3d.ops.ball_query",
                sources=["src/ball_query.cpp"],
            ),
            make_cpu_ext(
                name="knn_ext",
                module="mmdet3d.ops.knn",
                sources=["src/knn.cpp"],
            ),
            make_cpu_ext(
                name="assign_score_withk_ext",
                module="mmdet3d.ops.paconv",
                sources=["src/assign_score_withk.cpp"],
            ),
            make_cpu_ext(
                name="group_points_ext",
                module="mmdet3d.ops.group_points",
                sources=["src/group_points.cpp"],
            ),
            make_cpu_ext(
                name="interpolate_ext",
                module="mmdet3d.ops.interpolate",
                sources=["src/interpolate.cpp"],
            ),
            make_cpu_ext(
                name="furthest_point_sample_ext",
                module="mmdet3d.ops.furthest_point_sample",
                sources=["src/furthest_point_sample.cpp"],
            ),
            make_cpu_ext(
                name="gather_points_ext",
                module="mmdet3d.ops.gather_points",
                sources=["src/gather_points.cpp"],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )


