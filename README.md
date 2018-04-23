# CGDI-inpainting
M1 Computer Imaging project at ENS de Lyon

How to compile: (OpenCV3.1 or more recent and BoostOption must be installed)
	at the root, enter commande "cmake ."
	enter command "make"

The executable will be in directory bin.
To execute (at root): "bin/Inpainting [-options] [files]"
Type "bin/Inpainting --help" or "bin/Inpainting -h" for help/usage
2 basic usages:
	specifying a mask using option -m -> "bin/Inpainting -i image -m mask"
	no mask -> create a mask using the mouse, press Esc to run "bin/Inpainting -i image" (use +/- keys to adjust size)

To save the mask, use option -c or --create, specify a name (with the extension), the file "name.extension" will be created (default value if no name specified: "mask.png")
