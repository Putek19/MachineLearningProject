from setuptools import find_packages,setup

HYPEN_E_DOT = '-e .'

def get_requirements(file):
    '''
    this function return list of required packages
    '''
    requirements=[]
    with open(file) as file_obj:
        requirements = file_obj.readlines()
        [req.replace('\n',' ') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name = 'mlproject',
    version='0.0.1',
    author='Putek19',
    author_email='kubanowacki.jn@gmail.com',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')



)