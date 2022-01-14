import os

def get_phoneme(root):

    s = set({})

    for dir, _, files in os.walk(root):
        for file in files:
            if not file.endswith('.PHN'):
                continue
            phn_file = os.path.join(dir, file)
            fp = open(phn_file)
            for line in fp:
                ph = line.rstrip('\n').split()[2]
                s.add(ph)
            fp.close()

    return s


def main():
    print(get_phoneme('/tmp2/b08201047/data'))


if __name__ == "__main__":
    main()
