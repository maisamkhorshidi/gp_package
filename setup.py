from setuptools import setup, find_packages

# Function to read the requirements from requirements.txt
def read_requirements():
    with open('requirements.txt') as req_file:
        return req_file.readlines()

setup(
    name='genforge',
    version='0.1.0',
    packages=find_packages(),
    install_requires=read_requirements(),  # Read dependencies from requirements.txt
    author='Mohammad Sadegh Khorshidi',
    author_email='msadegh.khorshidi.ak@gmail.com',
    description='GenForge: Sculpting Solutions with Multi-Population Genetic Programming',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/maisamkhorshidi/genforge',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Development Status :: 3 - Alpha',
    ],
    python_requires='>=3.6',
)



