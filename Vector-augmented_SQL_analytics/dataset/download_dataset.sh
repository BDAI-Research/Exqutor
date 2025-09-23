
#!/bin/bash

dir=$(dirname "$0")

# Create DEEP directory if it doesn't exist
mkdir DEEP

# Download the file using axel if it doesn't exist
cd DEEP
if [ ! -f base.1B.fbin ]; then
    axel -n 10 https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
fi

# Download the query file if it doesn't exist
if [ ! -f query.public.10K.fbin ]; then
    axel -n 10 https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin
fi

# Create WIKI directory if it doesn't exist
cd "$dir"
mkdir WIKI

# Download the WIKI dataset parts if they don't exist
cd WIKI
for i in $(seq -w 00 09); do
    if [ ! -f wiki_all.tar.$i ]; then
        curl -O https://data.rapids.ai/raft/datasets/wiki_all/wiki_all.tar.$i
    fi
done

# Extract the WIKI dataset if not already extracted
if [ ! -d wiki_all_88M ]; then
    mkdir wiki_all_88M
    cat wiki_all.tar.* | tar -xf - -C wiki_all_88M/
fi


# # Create SIFT directory if it doesn't exist
# cd "$dir"
# mkdir SIFT

# # Download the SIFT file if it doesn't exist
# cd SIFT
# if [ ! -f base.1B.u8bin ]; then
#     axel -n 10 https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
# fi

# # Download the SIFT query file if it doesn't exist
# if [ ! -f query.public.10K.u8bin ]; then
#     axel -n 10 https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin
# fi

# Create SIMSEARCHNET directory if it doesn't exist
# cd "$dir"
# mkdir SIMSEARCHNET

# # Download the SIMSEARCHNET files if they don't exist
# cd SIMSEARCHNET
# if [ ! -f FB_ssnpp_database.u8bin ]; then
#     axel -n 10 https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/FB_ssnpp_database.u8bin
# fi

# if [ ! -f FB_ssnpp_public_queries.u8bin ]; then
#     axel -n 10 https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/FB_ssnpp_public_queries.u8bin
# fi


# add YFCC dataset
# clone big-ann-benchmarks repo and create symbolic link to dataset/yfcc
git submodule add https://github.com/harsha-simhadri/big-ann-benchmarks.git third_party/big-ann-benchmarks || true
git submodule update --init --recursive

# install requirements and create dataset
pip install -r third_party/big-ann-benchmarks/requirements.txt

# download yfcc dataset
cd third_party/big-ann-benchmarks
python create_dataset.py --dataset yfcc-10M
cd -

# create symbolic link
mkdir -p yfcc
ln -sfn $(pwd)/third_party/big-ann-benchmarks/data/yfcc $(pwd)/yfcc

cd "$dir"/third_party/tpch-kit/dbgen
make

./dbgen -f -s 100

cd "$dir"/third_party/tpcds-kit/tools
make

./dsdgen -SCALE 10 -DIR . >/dev/null 2>&1