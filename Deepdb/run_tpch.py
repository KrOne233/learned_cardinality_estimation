import argparse
import logging
import os
import shutil
import time

import numpy as np

from Deepdb.rspn.code_generation.generate_code import generate_ensemble_code
from Deepdb.data_preparation.join_data_preparation import prepare_sample_hdf
from Deepdb.data_preparation.prepare_single_tables import prepare_all_tables
from Deepdb.ensemble_compilation.spn_ensemble import read_ensemble
from Deepdb.ensemble_creation.naive import create_naive_all_split_ensemble, naive_every_relationship_ensemble
from Deepdb.ensemble_creation.rdc_based import candidate_evaluation
from Deepdb.evaluation.confidence_interval_evaluation import evaluate_confidence_intervals
from Deepdb.schemas.tpch.schema import gen_tpch_schema

dataset = 'tpch'

if dataset == 'tpch':

    os.makedirs('Deepdb/logs_deepdb', exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        # [%(threadName)-12.12s]
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("Deepdb/logs_deepdb/{}_{}.log".format(dataset, time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    # Generate schema
    csv_path = 'data/tpch'
    table_csv_path = csv_path + '/{}.csv'
    schema = gen_tpch_schema(table_csv_path)

    # Generate HDF files for simpler sampling
    hdf_path = 'data/tpch/hdf_files'
    max_rows_per_hdf_file = 100000000

    logger.info(f"Generating HDF files for tables in {csv_path} and store to path {hdf_path}")

    if os.path.exists(hdf_path):
        logger.info(f"Removing target path {hdf_path}")
        shutil.rmtree(hdf_path)

    logger.info(f"Making target path {hdf_path}")
    os.makedirs(hdf_path)

    prepare_all_tables(schema, hdf_path, csv_seperator=',', max_table_data=max_rows_per_hdf_file)

    logger.info(f"Files successfully created")

    # Generate sampled HDF files for fast join calculations
    hdf_sample_size = 10000
    logger.info(f"Generating sampled HDF files for tables in {csv_path} and store to path {hdf_path}")
    prepare_sample_hdf(schema, hdf_path, max_rows_per_hdf_file, hdf_sample_size)
    logger.info(f"Files successfully created")

    # Generate ensemble for cardinality schemas
    ensemble_strategy = 'rdc_based'
    ensemble_path = 'data/tpch/ensembles'
    samples_per_spn = [10000000, 10000000, 1000000, 1000000, 1000000, 100000, 100000]
    bloom_filters = False
    rdc_threshold = 0.3
    post_sampling_factor = [10, 10, 5, 5, 1, 1, 1]
    incremental_learning_rate = 0

    if not os.path.exists(ensemble_path):
        os.makedirs(ensemble_path)

    if ensemble_strategy == 'single':
        create_naive_all_split_ensemble(schema, hdf_path, samples_per_spn[0], ensemble_path,
                                        dataset, bloom_filters, rdc_threshold,
                                        max_rows_per_hdf_file, post_sampling_factor[0],
                                        incremental_learning_rate=incremental_learning_rate)
    elif ensemble_strategy == 'relationship':
        naive_every_relationship_ensemble(schema, hdf_path, samples_per_spn[1], ensemble_path,
                                          dataset, bloom_filters, rdc_threshold,
                                          max_rows_per_hdf_file, post_sampling_factor[0],
                                          incremental_learning_rate=incremental_learning_rate)
    elif ensemble_strategy == 'rdc_based':

        samples_rdc_ensemble_tests = 10000
        database_name = 'tpch'
        ensemble_budget_factor = 5
        ensemble_max_no_joins = 2
        pairwise_rdc_path = 'data/tpch/ensembles/pairwise_rdc.pkl'
        incremental_condition = None

        logging.info(
            f"maqp(generate_ensemble: ensemble_strategy={ensemble_strategy}, incremental_learning_rate={incremental_learning_rate}, incremental_condition={incremental_condition}, ensemble_path={ensemble_path})")

        candidate_evaluation(schema, meta_data_path=hdf_path, sample_size=samples_rdc_ensemble_tests,
                             spn_sample_size=samples_per_spn, max_table_data=max_rows_per_hdf_file,
                             ensemble_path=ensemble_path, physical_db_name=database_name,
                             postsampling_factor=post_sampling_factor,
                             ensemble_budget_factor=ensemble_budget_factor, max_no_joins=ensemble_max_no_joins,
                             rdc_learn=rdc_threshold,
                             pairwise_rdc_path=pairwise_rdc_path, rdc_threshold=0.15, random_solutions=10000,
                             bloom_filters=False, incremental_learning_rate=0, incremental_condition=None)

    else:
        raise NotImplementedError

    # cardinality prediction using ensemble generated
    from Deepdb.evaluation.cardinality_evaluation import evaluate_cardinalities

    ensemble_location = 'data/tpch/ensembles/ensemble_join_2_budget_5_10000000.pkl'
    query_file = 'data/tpch/deepdb_sql.sql'
    true_cardinalities_path = 'data/tpch/deepdb_true_cardinalities.csv'
    target_csv_path = 'results/result_deepdb_tpch.csv'
    pairwise_rdc_path = 'data/tpch/ensembles/pairwise_rdc.pkl'
    evaluate_cardinalities(ensemble_location=ensemble_location, query_filename=query_file,
                           target_csv_path=target_csv_path,
                           schema=schema, use_generated_code=False, physical_db_name='tpch', rdc_spn_selection=True,
                           pairwise_rdc_path=pairwise_rdc_path, true_cardinalities_path=true_cardinalities_path,
                           max_variants=1, merge_indicator_exp=False, exploit_overlapping=False, min_sample_ratio=0)

    for j in range(0, 7):
        ensemble_location = 'data/tpch/ensembles/ensemble_join_2_budget_5_10000000.pkl'
        query_file = f'data/tpch/more_joins/tpch_sql_test_{j}_deepdb_sql.sql'
        true_cardinalities_path = f'data/tpch/more_joins/tpch_sql_test_{j}_deepdb_true_cardinalities.csv'
        target_csv_path = f'results/result_deepdb_tpch_{j}.csv'
        pairwise_rdc_path = 'data/tpch/ensembles/pairwise_rdc.pkl'
        evaluate_cardinalities(ensemble_location=ensemble_location, query_filename=query_file,
                               target_csv_path=target_csv_path,
                               schema=schema, use_generated_code=False, physical_db_name='tpch', rdc_spn_selection=True,
                               pairwise_rdc_path=pairwise_rdc_path, true_cardinalities_path=true_cardinalities_path,
                               max_variants=1, merge_indicator_exp=False, exploit_overlapping=False, min_sample_ratio=0)
