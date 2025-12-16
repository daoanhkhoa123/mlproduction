import subprocess
from configs import CloudSetting

cloudSetting = CloudSetting() # type: ignore

def main():
    for d in cloudSetting.cloud_dependencies:
        subprocess.run(["uv", "add", d], check=True)

if __name__ == "__main__":
    main()