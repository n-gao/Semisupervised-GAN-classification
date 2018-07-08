@echo off
echo Downloading external wheels for win64...
If Exist "%TEMP%\scandir-1.6-cp36-cp36m-win_amd64.whl" echo Using cached file.
If Not Exist "%TEMP%\scandir-1.6-cp36-cp36m-win_amd64.whl" powershell -command "& {&'iwr' -outf %TEMP%\scandir-1.6-cp36-cp36m-win_amd64.whl https://download.lfd.uci.edu/pythonlibs/rqr3k8is/scandir-1.6-cp36-cp36m-win_amd64.whl}"
If Exist "%TEMP%\scipy-1.0.0-cp36-cp36m-win_amd64.whl" echo Using cached file.
If Not Exist "%TEMP%\scipy-1.0.0-cp36-cp36m-win_amd64.whl" powershell -command "& {&'iwr' -outf %TEMP%\scipy-1.0.0-cp36-cp36m-win_amd64.whl https://download.lfd.uci.edu/pythonlibs/rqr3k8is/scipy-1.0.0-cp36-cp36m-win_amd64.whl}"
If Exist "%TEMP%\pycosat-0.6.3-cp36-cp36m-win_amd64.whl" echo Using cached file.
If Not Exist "%TEMP%\pycosat-0.6.3-cp36-cp36m-win_amd64.whl" powershell -command "& {&'iwr' -outf %TEMP%\pycosat-0.6.3-cp36-cp36m-win_amd64.whl https://download.lfd.uci.edu/pythonlibs/rqr3k8is/pycosat-0.6.3-cp36-cp36m-win_amd64.whl}"
If Exist "%TEMP%\PyYAML-3.12-cp36-cp36m-win_amd64.whl" echo Using cached file.
If Not Exist "%TEMP%\PyYAML-3.12-cp36-cp36m-win_amd64.whl" powershell -command "& {&'iwr' -outf %TEMP%\PyYAML-3.12-cp36-cp36m-win_amd64.whl https://download.lfd.uci.edu/pythonlibs/rqr3k8is/PyYAML-3.12-cp36-cp36m-win_amd64.whl}"
echo Installing external wheels for win64...
pip install %TEMP%\scandir-1.6-cp36-cp36m-win_amd64.whl
pip install %TEMP%\scipy-1.0.0-cp36-cp36m-win_amd64.whl
pip install %TEMP%\pycosat-0.6.3-cp36-cp36m-win_amd64.whl
pip install %TEMP%\PyYAML-3.12-cp36-cp36m-win_amd64.whl
rem del /F /S /Q *.whl > NUL 2> NUL
echo Installing remaining requirements...
pip install -r requirements.txt
echo Installing conda...
pip install conda==4.2.7
echo Downloading pytorch... (this may take a while)
If Exist "%TEMP%\pytorch-0.2.1-py36he6bf560_0.2.1cu80.tar.bz2" echo Using cached file.
If Not Exist "%TEMP%\pytorch-0.2.1-py36he6bf560_0.2.1cu80.tar.bz2" powershell -command "& {&'iwr' -outf %TEMP%\pytorch-0.2.1-py36he6bf560_0.2.1cu80.tar.bz2 https://share.spreens.de/share/pytorch-0.2.1-py36he6bf560_0.2.1cu80.tar.bz2}"
echo Installing pytorch...
conda install %TEMP%\pytorch-0.2.1-py36he6bf560_0.2.1cu80.tar.bz2
rem del /F /S /Q pytorch*.tar.bz2 > NUL 2> NUL
echo Installing final requirements...
pip install -r requirements-after-torch.txt
