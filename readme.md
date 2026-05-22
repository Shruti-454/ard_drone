#clear the terminal first
STEP 1: 
python -m pip install --upgrade pip

STEP 2:
python -m pip install --upgrade --force-reinstall setuptools

STEP 3:
python -c "import pkg_resources; print('WORKING')"

STEP 4:
python tracker.py

else:(pkg_resources module actually install hi nahi hua abhi tak, even after setuptools attempt.)

python -m ensurepip --upgrade
python -m pip install setuptools==69.5.1
python -c "import pkg_resources; print('WORKING')"
python tracker.py

if these conditions satisfiess:-----
✅ ESP32-CAM stream URL is correct
✅ Both devices are on same network
✅ Your code compiles successfully
✅ OpenCV connects successfully
then your laptop will show the LIVE STREAM window automatically.


When working properly you should see:

✅ Live video
✅ Bounding boxes
✅ IDs moving with people
✅ Mouse click target locking
✅ Real-time tracking