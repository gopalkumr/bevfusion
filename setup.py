import os
from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="mmdet3d",
        packages=find_packages(),
        include_package_data=True,
        package_data={"mmdet3d.ops": ["*/*.so"]},
        classifiers=[
            # ... (classifiers can remain the same)
        ],
        license="Apache License 2.0",
        ext_modules=[],
        zip_safe=False,
    )
