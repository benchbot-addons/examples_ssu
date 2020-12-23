# Use a base image which provides as many of your dependencies as possible
FROM benchbot/submission:base

# Install votenet from Facebook Research
RUN git clone https://github.com/facebookresearch/votenet && \
    pip3 install 'torch<1.5.0' torchvision tensorflow matplotlib opencv-python \
    plyfile 'trimesh>=2.35.39,<2.35.40' 'networkx>=2.2,<2.3'

RUN apt update && apt install unzip && pushd votenet && wget \
    "https://drive.google.com/uc?export=download&id=1oem0w5y5pjo2whBhAqTtuaYuyBu1OG8l" \
    -O demo_files.zip && unzip demo_files.zip && rm demo_files.zip
    
# Install any extra dependencies
RUN pip3 install scipy


# Add the files from your solution
ADD ./semantic_slam_attempt .
ADD ./votenet_benchbot.py .

# Make the command used to run your solution the default command
CMD pushd votenet/pointnet2 && python3 setup.py install && popd && ./semantic_slam_attempt
