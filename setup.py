from setuptools import setup, find_packages

setup(
    name='e3sm_co2_diag',
    version='0.0a1',
    description='Package of tools for evaluating CO2 in E3SM',
    author='Daniel E. Kaufman',
    author_email='daniel.kaufman@pnnl.gov',
    packages=find_packages(),  #['e3sm_co2_diag'],
    include_package_data=True
    )
