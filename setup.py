from typing import List
from setuptools import setup, find_packages



# Creating a constant to remove -e .
HYPHEN_E_DOT = '-e .'

# Creating a function to fetch all packages from requirements.txt
def get_requirements(file_path:str)->List[str]:
    '''
    This function fetches all packages that are listed in requirements.txt
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        
    return requirements
           

# Creating the setup to install applicable packages
setup(
    name='score_preds',
    author='Abhijit Majumdar',
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)