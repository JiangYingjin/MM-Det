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
