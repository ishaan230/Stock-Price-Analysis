#!/bin/sh

file=$(fdfind . dataset | fzf)

python3 main.py $file
