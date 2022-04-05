mkdir -p data
cd data
# downloading dataset from https://commonvoice.mozilla.org/ru/datasets

# 811 MB (31 h)
wget https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-3/ru.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQ3GQRTO3KLTQPMPV%2F20220405%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220405T150524Z&X-Amz-Expires=43200&X-Amz-Security-Token=FwoGZXIvYXdzEAgaDNg4CLrdbrncQ3mO9iKSBFBMcnM1HDe3PFB%2B8S6zLdQQmg2jmuEWxai0TbyE4bNHtOM%2FKr%2FHTwYCX%2B3uLL4AvrG71eExVDjeiVNU0lmlQDUeNmJlOzmTMo%2FkZ2UVq8DY6qsUy2qCUm6UIwUXwSMIlfYPO8mgddEhaaRpo6VOjnqeqyfix5THdWVUQ3ScoPy0up6YDVSH3deaze8nqeUGRWY93bnM9cRlX2c0KYKUkbDciIJ7mNmtdfZpBaF3DpRemg1znaocbFhR5rL7ubzAXnDvE889iVLFNO9n7ttXWYUPxhKQTlG7YZ5ruXm8kFoG6iJJ%2Buv8eGZemDa2hejEoZIGMTFFm7zKbIKiFBx3rVy2WtaAte%2FpKy5HcB4VxuLdK4vR%2B0qU2OAvq35kklYqoDXl%2Bd0f3ZKn%2BUlQpjuzh0kfUd1Cnh%2FQs8gwarhttr%2F5NkojkAg8nBIyzXBUJ%2Fo2RUiHxrM8y2cnN0rDfOJAZ40Ks8Zg0V60rTTmWXokGG3QVT69BTEknKrgJ6O4u6hwP2HoLeTQoO8Za1KRtgvlwjF23SU88c7rDYeIU7K4LmbBRvwrcSrtuY%2BgzWjOQIc%2FMBGzKtivzSfsM5xrPD65npen7lAsenRe%2Bkk9pXt2fLJ34qM%2BBb6az87SE2r7u9fC4HSAEmnUXe24HFpzo3ni9Nij33jHjvdFfzV%2BVBv%2BFY1kn9zcD00ayoHAhJO3DE1Uja0uKJunsZIGMio00JzqUtagnr2cHa%2FdjUyyz6VGLhucSvVlUyzwOvoymR7bpFrrQ8SdGJA%3D&X-Amz-Signature=c5f7ce7cb04455ffab24bbe79a3a606f997d51fa362a2f704538fd3f54070934&X-Amz-SignedHeaders=host

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
