import setuptools

setuptools.setup(name='appliedvibration',
                 version = '0.12',
                 description='PoC applied vibration analysis tools',
                 url = '#',
                 author='RobotSquirrel',
                 install_requires=['opencv-python', 'numpy', 'matplotlib', 'scipy'],
                 author_email='',
                 packages=setuptools.find_packages(),
                 zip_safe=False)
