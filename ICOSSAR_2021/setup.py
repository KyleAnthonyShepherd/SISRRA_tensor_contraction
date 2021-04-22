from setuptools import setup, find_packages
import os
import sys
import datetime
# date=datetime.date.today().timetuple()
# version='1.'+str(date.tm_year-2019).zfill(2)+'.'+str(date.tm_yday).zfill(3)
# pip install wheel
# pip install pipreqs
# pip install pdoc
# git init

PackageName='ICOSSAR_2021'
pipignore=['p17']

if __name__ == "__main__":
    # if the command "python setup.py bdist_wheel" is run,
    # to generate the distribution wheel
    if len(sys.argv)==2:
        date=datetime.date.today().timetuple()
        version='1.'+str(date.tm_year-2019)+'.'+str(date.tm_yday)
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')
        command='pipreqs --force ../'+PackageName
        for ign in pipignore:
            command=command+' --ignore '+ign
        ErrorCode=os.system(command)
        if ErrorCode==1:
            sys.exit()
        # open the requirements file, replacing version equality signs
        file=open('requirements.txt','r')
        requirements=file.readlines()
        for r in range(0,len(requirements)):
            requirements[r]=requirements[r][:-1].replace('==','>=')
        file.close()
        packages=find_packages()
        packages=[PackageName+'.'+p for p in packages]
        setup(name=PackageName,
              version=version,
              description='CodeBase for SISRRA projects',
              url='https://duenas-osorio.rice.edu/sisrra',
              author='Kyle Shepherd',
              author_email='kas20@rice.com',
              license='GNU',
              package_dir = {'': '..'},
              packages=packages,
              install_requires=requirements,
              zip_safe=False
            )
        os.system('pip install --no-deps --force-reinstall ./dist/'+PackageName+'-'+version+'-py3-none-any.whl')
        os.system('rm -vrf ./build ./*.pyc ./*.tgz ./*.egg-info')
        os.chdir('..')
        # result=os.system('pdoc --html --force '+str(PackageName)+' --output-dir '+str(PackageName)+'/documentation')
        result=os.system('pdoc --docformat numpy -o ./'+str(PackageName)+'/documentation ./'+str(PackageName))
        os.chdir(PackageName)
        if result==1:
            sys.exit('Documentation ERROR')
        file=open('commit_message','r')
        text=file.read()
        file.close()
        print(text)
        if text!='':
            os.system('git add .')
            os.system('git commit -m "'+str(text)+'"')
            git commit -m "first push to github"
            os.system('git push -u https://github.com/KyleAnthonyShepherd/SISRRA_tensor_contraction.git main')
        file=open('commit_message','w')
        file.write('')
        file.close()

    else:
        wheel=os.listdir('dist')[0]
        os.system('pip install --no-deps --force-reinstall ./dist/'+wheel)
        # pack=sys.argv[2]
        # sys.argv.remove(pack)
        # file=open(str(pack)+'/requirements.txt','r')
        # requirements=file.readlines()
        # for r in range(0,len(requirements)):
        #     requirements[r]=requirements[r][:-1].replace('==','>=')
        # file.close()
        # # print(pack)
        # packages=find_packages(pack)
        # packages=[pack+'.'+p for p in packages]
        # # print(packages)
        # # input()
        # setup(name=pack,
        #       version=version,
        #       description='CodeBase for HARVEY analysis',
        #       url='KyleAnthonyShepherd.com',
        #       author='Kyle Shepherd',
        #       author_email='KyleAnthonyShepherd@example.com',
        #       license='GNU',
        #       packages=packages,
        #       # packages=['HARVEY'],
        #       install_requires=requirements,
        #       zip_safe=False
        #     )
        # os.system('pip install --no-deps --force-reinstall ./dist/'+pack+'-'+version+'-py3-none-any.whl')
        # os.system('rm -vrf ./build ./*.pyc ./*.tgz ./*.egg-info')
        # result=os.system('pdoc --html --force '+str(pack)+' --output-dir HARVEY/results')
        # if result==1:
        #     sys.exit('Documentation ERROR')
