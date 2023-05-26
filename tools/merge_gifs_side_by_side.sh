#!/bin/bash
# $1 : first gif
# $2 : second gif
# $3 : output gif

mkdir first
cd first
convert $1 -coalesce +adjoin x%04d.gif
cd ..

mkdir second
cd second
convert $2 -coalesce +adjoin x%04d.gif
cd ..

mkdir concat
for filename in first/*
do
  filename=`basename $filename`
  montage -tile 2x1 -geometry 1120x840 first/$filename second/$filename concat/$filename
done
convert concat/* $3

rm -rf first
rm -rf second
rm -rf concat