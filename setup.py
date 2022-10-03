from distutils.core import setup
from setuptools import find_packages

setup(
    name="openhsv",
    version="0.5",
    author="Andreas M Kist",
    author_email="me@anki.xyz",
    license="GPLv3+",
    packages=find_packages(),
    install_requires=[
        "pyqtgraph>=0.10.0",
        "numpy",
        "numba",
        "pandas",
        "qdarkstyle",
        "qimage2ndarray",
        "flammkuchen",
        "pillow",
        "scikit-image",
        "imageio>=2.4",
        "imageio-ffmpeg",
        "sounddevice",
        "opencv-python",
        "pyqt5",
        "tqdm",
        "scikit-learn"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="digital endoscopy camera audio",
    description="A package to perform high-speed videoendoscopy in a research setting.",
    include_package_data=True,
)