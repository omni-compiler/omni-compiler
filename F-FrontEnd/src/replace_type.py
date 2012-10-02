#!/usr/bin/python

import sys, re

TYPE_ID_DICT = {}
TYPE_SERIAL_ID = 1

def replace_type_id(line):
    global TYPE_SERIAL_ID, TYPE_ID_DICT

    type_pat = re.compile(r'(type|ref)="([A-Z][0-9a-f]+)"')
    match = type_pat.search(line)
    while match:
        type_id = match.group(2)
        if not TYPE_ID_DICT.has_key(type_id):
            TYPE_ID_DICT[type_id] = str(TYPE_SERIAL_ID)
            TYPE_SERIAL_ID += 1
        line = type_pat.sub('%s="%s"' % (match.group(1), TYPE_ID_DICT[type_id],), line)
        match = type_pat.search(line)

    return line

def erase_time(line):
    return re.sub(r'time="[^"]+"', "", line)

def main():
    if len(sys.argv) == 1:
        data_input = sys.stdin
    elif len(sys.argv) == 2:
        data_input = file(sys.argv[1])
    else:
        print "%s [datafile]" % (sys.argv[0],)
        sys.exit()

    for line in data_input:
        line = replace_type_id(line)
        line = erase_time(line)
        print line,

if __name__ == '__main__':
    main()
