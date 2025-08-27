from setuptools import setup, find_packages

setup(
    name='anta_database',
    version="0.1",
    author="Antoine Hermant",
    author_email= "antoine.hermant@etik.com",
    url="https://github.com/antoinehermant/anta_database",
    description= "SQLite database for the AntArchitecture radar data",
    long_description="""""",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'matplotlib',
    ]
)
