import os
import signal
from time import sleep
from subprocess import Popen, PIPE
import random

def with_ipcluster(func):
    def wrapped(*args, **kwargs):
        if "ipcluster_execute" in kwargs.keys() \
            and not kwargs["ipcluster_execute"]:
            return func(*args, **kwargs)
        if "ipcluster_nproc" in kwargs.keys():
            nproc = kwargs["ipcluster_nproc"]
        else:
            nproc = 1
        if "ipcluster_timeout" in kwargs.keys():
            timeout = kwargs["ipcluster_timeout"]
        else:
            timeout = 100

        hash = random.getrandbits(128)
        profile_name="default_%032x" % hash
        command = ["ipcluster", "start", "--profile", profile_name, "--n", str(nproc)]
        try:
            print("starting ipcluster...")
            proc = Popen(command, stdout=PIPE, stderr=PIPE)
            i = 0
            while True:
                sleep(1)
                outs = proc.stderr.readline().decode("ascii")
                print(outs.replace("\n", ""))
                if "successfully" in outs:
                    break
                if i > timeout:
                    raise TimeoutError("ipcluster timeout")
                i = i + 1
            print("started.")
            res = func(*args, **kwargs, ipcluster_profile=profile_name)
        finally:
            print("terminating ipcluster...")
            os.kill(proc.pid, signal.SIGINT)
    return wrapped

