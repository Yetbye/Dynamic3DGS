"""
Setup script for Dynamic3DGS

This package provides the Dynamic3DGS implementation for dynamic scene
reconstruction using deformable 3D Gaussian splatting.
"""

from setuptools import setup, find_packages
import os


def read_file(filename):
    """Read a file and return its contents."""
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return f.read()


def get_version():
    """Get version from __version__.py or default to 0.1.0."""
    version_file = os.path.join(os.path.dirname(__file__), 'dynamic_3dgs', '__version__.py')
    if os.path.exists(version_file):
        with open(version_file) as f:
            exec(f.read())
            return locals().get('__version__', '0.1.0')
    return '0.1.0'


setup(
    name='dynamic-3dgs',
    version=get_version(),
    description='Deformable 3D Gaussian Splatting for Dynamic Scenes',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/dynamic-3dgs',
    license='MIT',
    packages=find_packages(include=['dynamic_3dgs', 'dynamic_3dgs.*']),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'torchaudio>=2.0.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'Pillow>=8.0.0',
        'opencv-python>=4.5.0',
        'imageio>=2.9.0',
        'scikit-image>=0.18.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
        'tqdm>=4.60.0',
        'tensorboard>=2.8.0',
        'wandb>=0.12.0',
        'pyyaml>=6.0',
        'h5py>=3.6.0'
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'black>=21.0.0',
            'flake8>=3.9.0',
            'mypy>=0.910',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0'
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0'
        ]
    },
    entry_points={
        'console_scripts': [
            'dynamic3dgs-train=dynamic_3dgs.scripts.train:main',
            'dynamic3dgs-eval=dynamic_3dgs.scripts.eval:main',
            'dynamic3dgs-vis=dynamic_3dgs.scripts.vis:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Graphics :: 3D Rendering',
    ],
    keywords=[
        '3d gaussian splatting',
        'dynamic scenes',
        'neural rendering',
        'computer vision',
        '3d reconstruction'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/dynamic-3dgs/issues',
        'Source': 'https://github.com/yourusername/dynamic-3dgs',
        'Documentation': 'https://dynamic-3dgs.readthedocs.io/',
    }
)