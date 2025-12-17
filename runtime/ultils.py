import getpass
# NOTE: will be installed in cloud runtime
import gdown # type: ignore
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
        file_path = gdown.download_folder(link)
        return file_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None