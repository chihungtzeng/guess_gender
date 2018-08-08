#!/bin/bash
for fname in $(ls *_utf8.txt); do
    python extract_names.py -s $fname
done
