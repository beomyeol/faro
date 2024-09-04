import concurrent.futures
from pathlib import Path

from parse_twitter_minute import parse as parse_twitter_min

if __name__ == "__main__":
    twitter_input_path = Path("/srv/local/bj2/twitter/archiveteam-twitter-stream-2018-04/2018/04")
    target_dir_paths = [
        twitter_input_path.joinpath("%02d" % day).as_posix() for day in range(13, 23)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        min_dicts = list(executor.map(parse_twitter_min, target_dir_paths))