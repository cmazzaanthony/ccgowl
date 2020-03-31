import pandas as pd
import os
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def transform_gene_subset_50(input_filepath, output_filepath):
    l1 = ['U22376', 'X59417', 'U05259', 'M92287', 'M31211', 'X74262', 'D26156', 'S50223', 'M31523',
          'L47738', 'U32944', 'Z15115', 'X15949', 'X63469', 'M91432', 'U29175', 'Z69881', 'U20998',
          'D38073', 'U26266', 'M31303', 'Y08612', 'U35451', 'M29696', 'M13792']

    l2 = ['M55150', 'X95735', 'U50136', 'M16038', 'U82759', 'M23197', 'M84526', 'Y12670', 'M27891',
          'X17042', 'Y00787', 'M96326', 'U46751', 'M80254', 'L08246', 'M62762', 'M28130', 'M63138',
          'M57710', 'M69043', 'M81695', 'X85116', 'M19045', 'M83652', 'X04085']

    df = pd.read_csv(input_filepath/'golub_et_al/data_set_ALL_AML_train.csv')
    df1 = [col for col in df.columns if "call" not in col]
    df = df[df1]

    gene_list = list()
    for gene in l1 + l2:
        gene_list.append(df[df['Gene Accession Number'].str.contains(gene)])

    df1 = pd.concat(gene_list)
    df1 = df1.T
    df1.to_csv(output_filepath/'golub_et_al/AML_ALL_reduced_50.csv')


def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    transform_gene_subset_50(input_filepath, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    input_filepath = os.environ.get("INPUT_FILEPATH")
    output_filepath = os.environ.get("OUTPUT_FILEPATH")

    main(project_dir/input_filepath, project_dir/output_filepath)
