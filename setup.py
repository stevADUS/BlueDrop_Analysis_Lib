from setuptools import setup, find_packages

setup(
    name='BlueDrop_Analysis_Lib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'joblib',
        'matplotlib',
        'numpy',
        'pandas',
        'plotly',
        'scipy'
        ''
        # Add other dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)