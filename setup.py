from setuptools import setup

setup(name='vid2vid',
      version='1.0',
      description='vid2vid made pip-installable',
      url='https://github.com/NVIDIA/vid2vid',
      author='Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, '
             'Andrew Tao, Jan Kautz, Bryan Catanzaro',
      packages=['vid2vid'],
      install_requires=[
          'torch',
          'torchvision'
          ],
      zip_safe=False)


print('Now you still have to download and install flownet. This is done '
      'by running `vid2vid/scripts/download_flownet2.py`. A flownet checkpoint'
      ' can be found at `https://github.com/NVIDIA/flownet2-pytorch`')
