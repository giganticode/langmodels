import os

from setuptools import setup, find_packages

root_package_name = 'langmodels'
training_package_name = f'{root_package_name}.training'


def readme():
    with open('README.md') as f:
        return f.read()


def version():
    with open(os.path.join(root_package_name, 'VERSION')) as version_file:
        return version_file.read().strip()


setup(name='giganticode-langmodels',
      version=version(),
      description='A toolkit for applying machine learning to large source code corpora',
      long_description=readme(),
      long_description_content_type="text/markdown",
      url='http://github.com/giganticode/langmodels',
      author='giganticode',
      author_email='hlibbabii@gmail.com',
      license='MIT',
      packages=find_packages(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.7',
          'Operating System :: POSIX :: Linux',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      python_requires='>=3.6',
      keywords='big large data source code corpus machine learning nlp '
               'pytorch torch fastai language modeling',
      install_requires=[
        'fastai>=1.0.57,<2',
        'codeprep>=1.0.0,<2',
        'future>=0.18.2,<0.19',
        'comet-ml>=3.0.2,<4',
        'flatdict>=3.4.0,<4',
        'retrying>=1.3.3,<2',
        'psutil>=5.6.7,<6',
        'tqdm>=4.39,<5',
        'jsons>=1.0.0,<2',
        'numpy>=1.17,<2',
        'appdirs>=1.4.3,<2',
        'Columnar>=1.3.1,<2',
        'requests>=2.22,<3',
        'pysftp>=0.2.9,<0.3',
        'semver>=2.9.0,<3',
        'jq>=0.1.6,<0.2',
      ],
      entry_points={
          'console_scripts': [
              f'langmodels = {training_package_name}.__main__:main'
          ]
      },
      include_package_data=True,
      zip_safe=False)
