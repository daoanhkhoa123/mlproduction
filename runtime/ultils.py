import getpass
import subprocess
from typing import Callable

def getpass_deco(message:str=""):
    def wrapper(fn:Callable):
        def fn2():
            arg = getpass.getpass(message)
            return fn(arg)
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