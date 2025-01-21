from setuptools import setup, find_packages

setup(
    name="pyvisim",
    version="0.1.3",
    author="Nhat Huy Vu",
    author_email="vunhathuy234@gmail.com",
    description="A Python library for image similarity analysis using Image Encoders and Neural Networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MechaCritter/Python-Visual-Similarity",
    license="MIT",
    packages=find_packages(include=["pyvisim", "pyvisim.*"]),
    include_package_data=True,  # Ensure package data is included
    package_data={
        "pyvisim": ["res/model_files/*.pkl", 'res/logging_config.yaml'],  # Include only these files
    },
    python_requires=">=3.10",
    install_requires=[
        "h5py",
        "joblib",
        "matplotlib",
        "numpy",
        "opencv-python",
        "pandas",
        "pyaml",
        "PyYAML",
        "scikit-learn",
        "scipy",
        "seaborn",
        "torch",
        "torchvision",
        "torchaudio",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
