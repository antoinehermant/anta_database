from setuptools import setup, find_packages

setup(
    name='anta_database',
    version="0.1.1",
    author="Antoine Hermant",
    author_email= "antoine.hermant@etik.com",
    url="https://github.com/antoinehermant/anta_database",
    description= "SQLite database for the AntArchitecture Community Data",
    license="MIT",
    long_description="""""",
    packages=find_packages(),
    package_data={
        'anta_database': ['database/plotting/*.pkl'],
    },
    install_requires=[
        'pandas',
        'pyproj',
        'matplotlib',
        'colormaps',
    ]
)
