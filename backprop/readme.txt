The executable testcounting.exe was build under 64-bit Windows Vista using
gcc from the cygwin command line. Cygwin is a UNIX-like development
shell for Windows which you can download from the internet.
testcounting.exe needs cygwin under 64-bit Windows in order to run it.
It is here as a convenience as it's expected that you'll have your own 
C development system and make your own executable from that either
using the Makefile or by bringing the source into some sort of project
supported by your development environment. You need only:
testcounting.c - this contains main
backprop.c - the neural network library
backprop.h - a header file included by both the above
Makefile - for building with if you have the make utility
To run the resulting executable you'd do:
./testcounting<Enter>
The output appears as text in a command shell.
To test saving the neural network to a file, run it as:
./testcounting -s<Enter>
and before exiting, this will save the trained neural network to the file 
testcounting_network.bkp in the current directory.
You can then load it in the next time by running it as:
./testcounting -l<Enter>
which will load trained neural network from the file 
testcounting_network.bkp from the current directory rather than do a 
bunch of training first.
