import argparse
from pathlib import Path


def remove_duplicates(file, output):
    if output is None:
        output = file

    seen = set()
    lines = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if line.startswith('#') or not stripped:
                lines.append(line)
                continue

            if stripped not in seen:
                seen.add(stripped)
                lines.append(line)

    with open(output, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Processed {len(lines)} lines.")
    print(f"Unique Sentences: {len(seen)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=Path, help='Input file')
    parser.add_argument('-o', '--output', type=Path, help='Output file')
    args = parser.parse_args()

    remove_duplicates(args.file, args.output)


if __name__ == "__main__":
    main()
