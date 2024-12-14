cmake -G Ninja -B build 
cmake --build ./build --config release 
cd build/test 
matrix-bench.exe 
cd ../..
pause 
