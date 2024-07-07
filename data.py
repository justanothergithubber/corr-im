"""Functions relating to specific data processing."""

from collections import Counter, defaultdict
from csv import writer
from pickle import dump, load

from config import AMAZON_ALL_GENRE_FILE, AMAZON_FILE, AMAZON_RAW_OUTPUTS


def make_genre_csv() -> None:
    """Create CSV listing each ASIN and their genres."""
    dvd_asin_file = AMAZON_RAW_OUTPUTS / "dvd_asin.bin"
    cur_asin: str = ""
    # Simple caching
    if dvd_asin_file.exists():
        with open(dvd_asin_file, mode="rb") as dvd_file_io:
            dvd_asins: set[str] = load(dvd_file_io)
    else:
        with open(dvd_asin_file, mode="wb") as dvd_file_io:
            dvd_asins = set()
            with open(AMAZON_FILE, mode="r", encoding="utf-8") as io_obj:
                for line in io_obj:
                    ls_ = line.split()
                    if line.startswith("ASIN: "):
                        cur_asin = ls_[1]
                    if line.startswith("  group: DVD"):
                        dvd_asins.add(cur_asin)
            dump(dvd_asins, dvd_file_io)

    all_genres: set[str] = set()
    genre_flag: bool = False
    genre_dict: defaultdict[str, set[str]] = defaultdict(set)
    with open(AMAZON_FILE, mode="r", encoding="utf-8") as amazon_io:
        for line in amazon_io:
            ls_ = line.split()
            if line.startswith("ASIN: "):
                cur_asin = ls_[1]
            if cur_asin in dvd_asins:
                if line.startswith("  categories: "):
                    genre_flag = True
                    genre_dict[cur_asin] = set()
                elif line.startswith("  reviews: "):
                    genre_flag = False
                if genre_flag:
                    bar_split = line.split("|")
                    # how genres are denoted
                    if (len(bar_split) > 3) and (bar_split[3] == "Genres[404276]"):
                        genre = bar_split[4].strip()
                        all_genres.add(genre)
                        if genre not in genre_dict[cur_asin]:
                            genre_dict[cur_asin].add(genre)

    all_genres_list = sorted(list(all_genres))
    num_genres = len(all_genres_list)
    genre_counter: Counter[str] = Counter()
    with open(
        AMAZON_ALL_GENRE_FILE,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as genre_io:
        csv_writer = writer(genre_io)
        csv_writer.writerow(["ASIN"] + all_genres_list)
        for asin, genre_set in genre_dict.items():
            to_write_list: list[str | int] = [""] * num_genres
            for genre in genre_set:
                genre_counter[genre] += 1
                to_write_list[all_genres_list.index(genre)] = 1
            csv_writer.writerow([asin] + to_write_list)
