import os

def running_environment():
    if os.environ["HOME"] == "/home/ec2-user":
        return "sagemaker"
    else:
        return "local"