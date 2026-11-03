# A-DAEC

Experimental implementation accompanying the paper “A-DAEC: Adaptive Duplicate-Aware Event Commitment for Efficient Blockchain-Based Auditing of IoT Intrusion Events.”

## Requirements

Python 3.11, pandas, and NumPy.

Install the required packages:

    pip install -r requirements.txt

SHA-256 operations use Python's built-in hashlib module.

## Dataset

Download train_test_network.csv from:

https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset

Place the CSV in the same directory as adaec_blockchain_experiment_v3.py.

The evaluated file contains 211,043 records and 44 columns, including 50,000 normal records and 161,043 attack records. The experiments use this file without additional record removal or class rebalancing.

## Running the experiment

    python adaec_blockchain_experiment_v3.py

The script evaluates Full Anchoring, Fixed-Batch, Class-Based aggregation, Exact-Signature DAEC (named DAEC V2 in the code), and A-DAEC (named A-DAEC V3 in the code).

Results are saved in the adaec_results_v3 directory. Existing result files with the same names are overwritten when the script is rerun.

## Experimental configuration

- Maximum group size: 100 events.
- Main semantic-homogeneity threshold: 90%.
- Sensitivity thresholds: 80%, 85%, 90%, and 95%.
- Timing repetitions: five per method and dataset scale.
- Verification-test samples: 1,000 events per method and dataset scale.

A-DAEC algorithm time includes semantic preparation plus grouping and commitment construction. Processing_Time_Mean_s excludes semantic preparation; Algorithm_Time_Mean_s includes it.

Verification_Time_ms measures proof generation, verification of the original hash, and verification of a deliberately modified hash.

The implementation uses a local hash-linked commitment mechanism. It does not deploy a blockchain network or measure consensus latency or transaction fees.# A-DAEC
Experimental implementation of A-DAEC for IoT intrusion-event auditing
