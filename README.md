flownet2 must be compiled. To be able to set proper host compiler with
environment variable `CC`, use the patched files under `patches`.
gcc-5 can be used on compgpu servers with

    export CC=/export/home/pesser/local-gcc-5/bin/gcc

To use the patched file simply copy each setup.py to its corresponding location.
The run
    bash install.sh

All the above steps need to be redone when running the ``download_flownet2.py`` script.
