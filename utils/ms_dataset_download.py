import subprocess
import sys


def install_dependencies():
    """安装必要的依赖包"""
    dependencies = ["modelscope", "datasets", "oss2", "addict"]

    for package in dependencies:
        try:
            # 尝试导入包以检查是否已安装
            __import__(package)
            print(f"{package} 已安装")
        except ImportError:
            print(f"正在安装 {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"{package} 安装成功")
            except subprocess.CalledProcessError as e:
                print(f"{package} 安装失败: {str(e)}")


install_dependencies()


from modelscope.msdatasets.download.download_manager import DataStreamingDownloadManager
from modelscope.msdatasets.download.download_config import DataDownloadConfig
from modelscope.hub.api import HubApi


def download(dataset: str, file_list: list[str], save_dir: str):
    download_config = DataDownloadConfig()
    namespace, dataset_name = dataset.split("/")
    download_config.cache_dir = save_dir
    download_config.oss_config = HubApi().get_dataset_access_config(
        dataset_name, namespace
    )
    dl_manager = DataStreamingDownloadManager(download_config=download_config)
    dst_file_list = dl_manager.download(file_list)


if __name__ == "__main__":
    download(
        dataset="cccnju/Gen-Video",
        file_list=["GenVideo-Val.zip"],
        save_dir="/root/data/GenVideo",
    )
