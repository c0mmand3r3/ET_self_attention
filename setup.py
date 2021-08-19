import os

from setuptools import setup, find_packages

_root = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_root, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

with open('HISTORY.rst', encoding="utf-8") as f:
    history = f.read()

version = {}
with open(os.path.join(_root, 'ET_self_attention', 'version.py')) as f:
    exec(f.read(), version)

requirements = []
try:
    with open('requirements.txt', encoding="utf-8") as f:
        requirements = f.read().splitlines()
except IOError as e:
    print(e)

test_requirements = []

setup(
    name='ET_self_attention',
    version=version['__version__'],
    description='E.T.: Re-Thinking Self-Attention for Transformer Models on GPUs',
    long_description=readme + "\n\n" + history,
    author='Shiyang Chen, Shaoyi Huang+, Caiwen Ding+ and Hang Liu',
    implemetation_author='Anish Basnet',
    url='',
    license='',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='Transformer, BERT, self-attention',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
