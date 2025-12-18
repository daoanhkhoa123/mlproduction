import getpass
import subprocess
from typing import Callable


def getpass_deco(message:str=""):
    def wrapper(fn:Callable):
        def fn2(*args, **kwargs):
            secret = getpass.getpass(message)
            return fn(secret, *args, **kwargs)
        return fn2
        
    return wrapper

@getpass_deco("Type in the google drive folder link:")
def gdown_folder(link:str):
    try:
        # gdown can handle both file IDs and full links
        subprocess.run(["gdown", "--folder", link], check=True)
        print("Folder downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return None
    
@getpass_deco("Type in the ngrok token:")
def ngrok_start(token:str, port:int=5000):
    try:
        subprocess.run(["ngrok", "config", "add-authtoken", token], check=True)
        subprocess.run(["ngrok", "http", str(port)], check=True)
    
    except subprocess.CalledProcessError as e:
        print("Ngrok failed:" , e)
