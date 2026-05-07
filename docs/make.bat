@ECHO OFF

set SPHINXBUILD=sphinx-build
set SPHINXAUTOGEN=sphinx-autogen
set SOURCEDIR=.
set BUILDDIR=_build
set APIDOCINDEX=%SOURCEDIR%\api\index.rst
set APIGENDIR=%SOURCEDIR%\api\generated

if "%1" == "" goto help

rmdir /s /q "%APIGENDIR%" 2>NUL
mkdir "%APIGENDIR%" 2>NUL
%SPHINXAUTOGEN% -t "%SOURCEDIR%\_templates" -o "%APIGENDIR%" "%APIDOCINDEX%"
for %%F in ("%APIGENDIR%\*.rst") do %SPHINXAUTOGEN% -t "%SOURCEDIR%\_templates" -o "%APIGENDIR%" "%%~fF"
for %%F in ("%APIGENDIR%\*.rst") do %SPHINXAUTOGEN% -t "%SOURCEDIR%\_templates" -o "%APIGENDIR%" "%%~fF"
%SPHINXBUILD% -b %1 %SOURCEDIR% %BUILDDIR%/%1
goto end

:help
echo Usage: make.bat [builder]
echo.
echo Example:
echo   make.bat html

:end
