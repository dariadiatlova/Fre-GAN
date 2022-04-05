mkdir -p data
cd data
# downloading dataset from https://commonvoice.mozilla.org/ru/datasets

# 811 MB (31 h)
wget "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-3/ru.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQ3GQRTO3ADW7SFMJ%2F20220405%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220405T155152Z&X-Amz-Expires=43200&X-Amz-Security-Token=FwoGZXIvYXdzEAkaDInURvpGCK9uy8bxlCKSBC2Ca54JDHBbqYP7utHZCUvrFTT3322SPqgDRuuqOIJjaY3bPQtPbgg2ilDBF0l%2BVV93NPRK8psGbm9hPRCmgACMm1%2FNav%2Bxdol%2Ftkd2TDaVmURnR4dwWe%2FGjCsefz3M5xhgogjdn%2BUXTNkIPFMlXXPWrdEu6vjfRIulV5B4F%2FsswhaBUD%2FpcVzRtt0yN89ea0DQspSPcpfrLshWJigxbqHoUm%2Bouz423kUOV%2FkMLJcoMJ8ifGHmbSUiaowG34ABBNuHcNMVVw3i1O2rdFR7IZIgrmx9d3esqSw7yypMs7wYE5r7KfypqKqNWY4ywcx9I7hWj6OIkCcOk0hUQnBjLqJFIBow1MKyyHTAPpcYfKxShZaHUb503hwhFhxYAwangINlgQQjVwQUfGTG%2FKsOdGQbK%2FPD%2F9l7u3ajJIwYGOlJJ6Nd18cz7iRsB97wky1tWwgl2St7f6j5WfcQRrEQ3QRBKcx%2BvNbcIm6lEk2YGX6Y6F0crzB0BuDWC8S2GS2zSEC%2F7bJHQ4vBaz5RRyNoChMaW4gxI0ZIcqbL5rln3o6pflW5rQjJiqTTWpBUvEXTFSGNPEfTuxHLh%2Fyq5%2BjCNJRJX3SL3w5fEYXcK0pKG9qVMTuo1PGckFmrqmBdvV0HX%2FbglpaXz0PIzng8jz29xVJaXatVK1EYKxQAPRwA1Nd%2Fiuw9UkJ5Ifunxh7ushDktJiGKKrCsZIGMipu7OnI2VBKRycCFuZg5ogwpEe8uwv%2FVhzn2KqLip7txleOyECs4VG3y%2Fo%3D&X-Amz-Signature=ebd021fcae9c73aac9b5e8433e851296a69246f35bd04d3c92ea8aafcd990a3a&X-Amz-SignedHeaders=host"
mv ru/train.tsv .
mv ru/test.tsv .

# 5 GB (130 h)
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
