from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name="usualsuspects",
    version='0.2.1',
    description="Some quick tools for generating visualisations that pop up in ML papers all the time",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/paulmorio/usualsuspects",
    author="Paul Scherer",
    author_email="paul.geo2dr@gmail.com",
    license="MIT",
    install_requires=['numpy', 'scikit-learn', 'tqdm', 'matplotlib'],
    packages=find_packages(),
    classifiers=['Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Programming Language :: Python :: 3.6',
                 'Operating System :: OS Independent'],
    zip_safe=False
)
