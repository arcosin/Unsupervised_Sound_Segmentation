from colorama import Fore
import pickle


def print_green(text: str) -> None:
    print(Fore.GREEN + text + Fore.RESET)


def print_red(text: str) -> None:
    print(Fore.RED + text + Fore.RESET)


def print_yellow(text: str) -> None:
    print(Fore.YELLOW + text + Fore.RESET)


def read_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data
