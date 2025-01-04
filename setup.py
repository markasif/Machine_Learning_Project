from setuptools import find_packages,setup
from typing import List

def get_requirement(file_path:str)->List[str]:
    '''
    this function return the list of requirement
    
    '''
    Hpen_Dot_E = "-e ."
    requirements=[]
    with open(file_path) as file_obj:
        requirements =  file_obj.readlines()
        requirements= [req.replace("\n"," ") for req in requirements]

        if Hpen_Dot_E in requirements:
            requirements.remove(Hpen_Dot_E)

    return requirements


setup(
    name="Machine_Learning_Project",
    version="0.0.1",
    author="Asif ap",
    author_email="apasif243@gmail.com",
    packages=find_packages(),
    install_requires = get_requirement('requirement.txt')

)