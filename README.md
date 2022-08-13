
## Requirements
* CMake
* libtorch
* OpenCV

## Building
### Linux
```
$ cmake .
$ make
```

## Running with rtsp stream
```
$ ./magnification levels=6 alpha=10 lambda_c=16 cutoff_frequency_high=0.4 cutoff_frequency_low=0.05 rtsp://192.168.1.1/camera
```
