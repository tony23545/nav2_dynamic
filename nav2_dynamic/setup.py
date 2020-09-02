from setuptools import setup, find_packages

package_name = 'nav2_dynamic'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tony',
    maintainer_email='csj15thu@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detectron_node = nav2_dynamic.detection.detectron_node:main',
            'hungarian_node = nav2_dynamic.tracking.hungarian_node:main'
        ],
    },
)
