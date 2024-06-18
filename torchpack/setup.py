from setuptools import find_packages, setup

with open('requirements.txt', 'r') as f:
    install_requires = [line for line in f]


def readme():
    with open('README.md') as f:
        content = f.read()
    return content


def get_version():
    version_file = 'torchpack/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


setup(
    name='torchpack',
    version=get_version(),
    description='A set of interfaces to simplify the usage of PyTorch',
    long_description=readme(),
    keywords='computer vision',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities',
    ],
    url='https://github.com/hellock/torchpack',
    author='Kai Chen',
    author_email='chenkaidev@gmail.com',
    license='MIT',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=install_requires,
    zip_safe=False
)  # yapf: disable