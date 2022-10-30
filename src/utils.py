from colorama import Fore


def print_green(text: str) -> None:
    print(Fore.GREEN + text + Fore.RESET)


def print_red(text: str) -> None:
    print(Fore.RED + text + Fore.RESET)


def print_yellow(text: str) -> None:
    print(Fore.YELLOW + text + Fore.RESET)
