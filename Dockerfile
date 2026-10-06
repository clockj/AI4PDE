# RealPDE Track 2 (LTTTA), team xtgg233: the training package's image.
# The Final phase starting kit's Dockerfile, unchanged: the base is the organizers' Track 2 evaluation
# image (PyTorch 2.2.2 + CUDA 12.1, Python 3.10, numpy 1.26.4, h5py 3.16.0), which already holds every
# package the training uses, so nothing is installed.
FROM w3nhao/realpde-track2@sha256:f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972
WORKDIR /workspace
COPY . /workspace
