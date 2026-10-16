#!/usr/bin/env bash
# Downloads the 300-d GloVe Common Crawl vectors used in the paper (~2 GB zip, ~5.6 GB unpacked).
set -e
cd "$(dirname "$0")/../data"
[ -f glove.840B.300d.txt ] && { echo "already present"; exit 0; }
wget -c https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip -o glove.840B.300d.zip && rm glove.840B.300d.zip
echo "done: data/glove.840B.300d.txt"
