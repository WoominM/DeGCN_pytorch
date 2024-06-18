if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
    pip install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl
elif [[ "$TRAVIS_PYTHON_VERSION" == "3.5" ]]; then
    pip install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
elif [[ "$TRAVIS_PYTHON_VERSION" == "3.6" ]]; then
    pip install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
fi
pip install torchvision