meson setup builddir
cd builddir
meson compile
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)
        machine="Linux";;
    MINGW*)
        machine="MinGw";;
esac

if [ "$machine" == "MinGw" ]; then
    mv src/conv_greed.exe ../..
    mv src/cvar_greed.exe ../..
    mv src/pmc_greed.exe ../..
    mv src/cvar_est.exe ../..
    mv src/pmc_est.exe ../..
elif [ "$machine" == "Linux" ]; then
    mv src/conv_greed ../..
    mv src/cvar_greed ../..
    mv src/pmc_greed ../..
    mv src/cvar_est ../..
    mv src/pmc_est ../..
fi
