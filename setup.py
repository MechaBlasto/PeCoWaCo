import setuptools
from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()


setup(
    name="pecowaco",
    version='1.0.0',
    author='Varun Kapoor,OOzge Ozguc',
    author_email='randomaccessiblekapoor@gmail.com',
    url='https://github.com/MechaBlasto/PeCoWaCo/',
    description='Analysis of Periodic contractions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "numpy",
        "pandas",
        "napari",
        "bokeh",
        "scikit-image",
        "scipy",
        "tifffile",
        "matplotlib",
        
    ],
    entry_points = {
        'console_scripts': [
            'track = pecowaco.__main__:main',
        ]
    },
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
    ],
)
