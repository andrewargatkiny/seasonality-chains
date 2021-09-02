from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')
setup(
    name='seasonality-chains',
    version='0.2.0',
    description='A framework for forecasting time-series with multiple seasonal components',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/andrewargatkiny/seasonality-chains',
    author='Andrew Argatkiny',
    author_email='andrewkrskde@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    keywords = ('time series, forecast, predict, analysis'
    'seasonality, statistics, econometrics'),
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'scikit-learn'],
    python_requires='>=3.6'
)