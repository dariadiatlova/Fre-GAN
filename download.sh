mkdir -p data
cd data
# downloading dataset from https://commonvoice.mozilla.org/ru/datasets

# 811 MB (31 h)
wget -O ru.tar "$1"
tar -xvsf ru.tar

mv ru/train.tsv .
mv ru/test.tsv .

# 5 GB (193 h)
#wget https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-8.0-2022-01-19/cv-corpus-8.0-2022-01-19-ru.tar.gz
#tar -xvsf cv-corpus-8.0-2022-01-19-ru.tar.gz

#mv cv-corpus-8.0-2022-01-19/ru/train.tsv .
#mv cv-corpus-8.0-2022-01-19/ru/test.tsv .

# downloaded data is in .mp3 here we convert it into wav
mkdir -p audio

# write to an array all paths to mp3 files
#FILES=$(find cv-corpus-8.0-2022-01-19/ru/clips -type f -name "*.mp3")
FILES=$(find ru/clips -type f -name "*.mp3")

# multiprocess implementation of file conversion
mp3_to_wav() {
  f=$1
  filename="${f##*/}"
  filename="${filename%.*}"
  ffmpeg -i "$f" "audio/${filename}.wav";
}

# make a constrain on max number of processes to use
max_num_processes=$(ulimit -u)
limiting_factor=4
num_processes=$((max_num_processes/limiting_factor))

# loop trough all mp3 files and convert them to wav using multiprocessing
for f in $FILES; do
  ((i=i%num_processes)); ((i++==0)) && wait
  mp3_to_wav "$f" &
  done
