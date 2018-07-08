@echo off
echo Creating virtualenv...
python.exe -m pip install virtualenv
cd ..
rmdir /S /Q .venv > NUL 2> NUL
python.exe -m virtualenv --no-site-packages .venv
call .venv\Scripts\activate.bat & ^
copy /Y .venv\Scripts\activate.bat .venv\Scripts\activate.bat.bkp > NUL 2> NUL & ^
copy /Y .venv\Scripts\activate .venv\Scripts\activate.bkp > NUL 2> NUL & ^
cd setup & ^
call install.bat & ^
cd .. & ^
move /Y .venv\Scripts\activate.bat.bkp .venv\Scripts\activate.bat > NUL 2> NUL & ^
move /Y .venv\Scripts\activate.bkp .venv\Scripts\activate > NUL 2> NUL & ^
cd setup
