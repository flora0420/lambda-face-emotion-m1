# Deployment on EC2

We are going to deploy a pretrained image segmentation model:

https://github.com/tensorflow/models/tree/master/research/deeplab

## Create EC2 Instance

- Go to EC2 console: https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#
- Create EC2 instance
- Pick amazon linux
- Pick instance type: At least t3.medium
- Create key-pair
- Download key
- Edit network
- Enable IPV4 address
- Open port 8000 from anywhere
- Launch Instance

## Install dependencies
- Get the ip address of the instance
- Change key permissions to 400 (`chmod 400 key.pem`)
- SSH into the machine `ssh -i key.pem ec2-user@ec2.ip.address`
- Install git if needed (`sudo apt install git` for ubuntu based distros, `sudo yum install git` for amazon linux)
- Install pip if needed (`sudo apt install python3-pip` for ubuntu based distros, `sudo yum install python3-pip` for amazon linux)
- Clone the repo (`git clone ...`)
- If there's permission issues with gitlab, generate ssh keys (`ssh-keygen`) and add them to github account
- CD into the folder (`cd cloned-repo`)
- Install the requirements (`pip3 install -r requirements.txt`)

## Run API
- Run the api (`uvicorn app:app --host 0.0.0.0 --port 8000`)
- Create a request with docs (http://ec2.ip.address:8000/docs)

# Run API locally

Read this section if you are on an M1 chip ( arm64 / silicon; in theory it should work on M2 as well, yet to be tested ) and would like to launch the API locally. This part has been tested on an Apple M1 chip, macOS Monterey 12.6 machine. 

## Install Tensorflow for M1
Follow instructions step by step from [Metal: Getting Started with tensorflow-metal PluggableDevice](https://developer.apple.com/metal/tensorflow-plugin/), please read the NOTEs carefully and apply them accordingly. 

After you install `miniforge3`, you can use it to create a new environment, say `tf-metal` and then install the `tensorflow-deps` to the environment.

After following these three steps, make sure in your terminal to check if tensorflow is installed correctly by running 

```
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

<details>
<summary> Example Output 

    Metal device set to: Apple M1

    systemMemory: 8.00 GB
    maxCacheSize: 2.67 GB

    2022-09-28 20:46:17.797673: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
    2022-09-28 20:46:17.797813: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
    tf.Tensor(-692.91614, shape=(), dtype=float32)

</summary>
</details>

If this is NOT successful ( symptoms: your python crashes ), please reach out to your instructors.

## Install dependencies

1. install the dependencies from `requirements-notf.txt`

    ```
    python -m pip install -r requirements-notf.txt
    ```

    You will not need `tensorflow` here, since it has been installed in the previous step. 

2. Troubleshooting: 
    Fire up a Python kernel from terminal and simply import the following packages as shown in `deeplab.py` to make sure they are installed correctly. 

    ```
    import cv2
    import tensorflow as tf
    ```

    a. Another difference you notice in `requirements-notf.txt` is that we are using `opencv-python` instead. This is to avoid a conflict in `numpy` versions: if you see an `ImportError: numpy.core.multiarray failed to import`, that's due to the conflict and installing the correct version of `opencv-python` should fix it.

    b. Another possible error you might see is `AssertionError: Duplicate registrations for type 'experimentalOptimizer'`, not sure about the exact problem, yet uninstall `keras` shall solve it: 
    
        ```
        python -m pip uninstall keras
        ```
5. Run the API locally

    ```
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```

    Your API will be running at `http://localhost:8000/docs`
    