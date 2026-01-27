def keep_last_n_lines(filename, n=1000):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines[-n:])

if __name__ == "__main__":
    keep_last_n_lines("ragnet/outputs.txt")

