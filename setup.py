from setuptools import setup, find_packages

setup(
    name="deep_feature_explorer",
    version="0.0.1",
    author="Param Uttarwar",
    author_email="param.uttarwar@casablanca.ai",
    description="A general purpose tools to visualize neural networks",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pfgurus/NetworkVisualizer",  # URL to your project
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "dfe": ["*.txt",],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[ 'PyQt5', 'opencv-python-headless','torch'],

)
