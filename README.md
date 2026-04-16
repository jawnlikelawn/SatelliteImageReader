# SatelliteImageReader

This doesnt work very well! 

You can get .fits files from https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html, or anywhere else .fits files are found. 

Put the fits file into "FITS_FILES", this makes them viewable and queue-able in FITS_VIEWER.py. 

Open and run FITS_VIEWER.py, an interface should open, select your .fits file to view it. You can rename and queue it or export it as a png using the top menu. 

To identify objects we'll have to prepare the ML classifier. 

Run generate_data, it will make a two folders of automated training data "ships" and "not ships". Use these folders to drop positive training data into later. 

Run train.py to train the classifier. it will try and id ships and not ships. 

Run scan to scan all the images in FILES_TO_SCAN
