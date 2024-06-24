def read_lines(file_path, encoding='utf-8'):
    with open(file_path, encoding=encoding) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines
