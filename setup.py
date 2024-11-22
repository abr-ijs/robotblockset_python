from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name="robotblockset",
    version="1.0.0",
    description=("Robotblockset (RBS) allows control of various robots and provides common functions such as transformations."),
    url="https://repo.ijs.si/leon/robotblockset_python",
    author="Leon Zlajpah",
    author_email="leon.zlajpah@ijs.si",
    license="MIT",
    packages=["robotblockset", "robotblockset.ros"],
    install_requires=["numpy", "quaternionic", "matplotlib"],
    keywords="robot toolbox python robots control transformations trajectory",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
