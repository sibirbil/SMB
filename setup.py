from setuptools import setup

setup(name='SMB',
      version='0.1.0',
      description='Sochastic gradient descent with model building',
      url='git@github.com:sibirbil/SMB.git',
      maintainer='Ozgur Martin',
      maintainer_email='ozgurmartin@gmail.com',
      license='MIT',
      packages=['smb'],
      zip_safe=False,
      install_requires=[
        'torch>=1.0.0'
        'torchvision>=0.10.0'
      ]),
