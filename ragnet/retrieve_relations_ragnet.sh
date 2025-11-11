#!/bin/sh
set -e # Exit if any command fails


# Encode relations using the trained bi_encoder and build index of encoded relations
python3 relation_retrieval/bi_encoder/build_and_search_index.py encode_relation --dataset WebQSP
python3 relation_retrieval/bi_encoder/build_and_search_index.py build_index --dataset WebQSP

# Check if encoded relation index file exists
if [ -f "data/WebQSP/relation_retrieval/bi_encoder/index/rich_relation_3epochs/ep_3_flat.index" ]; then
    echo "Relation index file exists."
else
    echo "Relation index file not found! Exiting..."
    exit 1
fi

# Encode questions using the trained bi_encoder and retrieve indexed relations
python3 relation_retrieval/bi_encoder/build_and_search_index.py encode_question --dataset WebQSP --split test
python3 relation_retrieval/bi_encoder/build_and_search_index.py retrieve_relations --dataset WebQSP --split test

# Check if cross_encoder data file exists
if [ -f "data/WebQSP/relation_retrieval/cross_encoder/rich_relation_3epochs_question_relation/WebQSP.test.tsv" ]; then
    echo "Cross-encoder data file exists."
else
    echo "Cross-encoder data file not found! Exiting..."
    exit 1
fi

# Run cross_encoder to rank retrieved relations
#sh scripts/run_cross_encoder_WebQSP_question_relation.sh predict rich_relation_3epochs_question_relation test WebQSP_ep_3.pt

# Check if cross_encoder output file exists
if [ -f "data/WebQSP/relation_retrieval/cross_encoder/saved_models/rich_relation_3epochs_question_relation/WebQSP_ep_3.pt_test" ]; then
    echo "Cross-encoder output file exists."
else
    echo "Cross-encoder output file not found! Exiting..."
    exit 1
fi

# Saved sorted relations
python3 data_process.py merge_relation --dataset WebQSP --split test

# Check if final candidate relations file exists
if [ -f "data/WebQSP/relation_retrieval/candidate_relations/WebQSP_test_cand_rels_sorted.json" ]; then
    echo "Final candidate relations file exists."
else
    echo "Final candidate relations file not found! Exiting..."
    exit 1
fi

