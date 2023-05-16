#!/usr/bin/env python3

# Cross-match of the optical nnmag counterparts obtained in lh_class.py with
# the SRGz catalog to get SRGz features. Merging nnmag and SRGz catalog to
# get two versions of optical counterparts (nnmag and SRGz) for the every
# X-ray source.

import os
import pandas as pd
from lh_class import lh_functions as lhf
pd.options.mode.chained_assignment = None
from pathlib import Path
import argparse
from astropy.coordinates import SkyCoord

INPUT_HELPSTR = 'Please enter the path to the input data. \
It can be downloaded here: https://disk.yandex.ru/d/UybgtHAwTIdaWA'
OUTPUT_HELPSTR = 'Please enter the path to the output directory.'

parser = argparse.ArgumentParser()  # create a parser

# specify the arguments
# TODO: make shure that the link is correct before the release
parser.add_argument(
    '-input_path', '-i', type=str, required=True, help=INPUT_HELPSTR
    )
parser.add_argument(
    '-output_path', '-o', type=str, required=True, help=OUTPUT_HELPSTR
    )

args = parser.parse_args()
input_dir_path = args.input_path

# path to save the table of sources matched with external catalogs
# and classified as extragalactic/not extragalactic
# this is mediate (auxiliary) data
mediate_dir_path = args.output_path + 'mediate_data/'
# create saving directory if it doesn't exist
Path(mediate_dir_path).mkdir(parents=True, exist_ok=True)

SRGZ_FILENAME = 'lhpv_03_23_sd01_a15_g14_srgz_CatA_XnX_model4_SQG_model5_v20221207'
srgz_cat_path = args.input_path + SRGZ_FILENAME

# path to the matched and classified catalog
matched_class_path = mediate_dir_path + 'matched_and_classified.gz_pkl'

save_directory = os.path.dirname(matched_class_path)
# create saving directory if it doesn't exist
Path(save_directory).mkdir(parents=True, exist_ok=True)

srgz_prepared_path = save_directory + '/srgz_nnmag.gz_pkl'


def main():

    print()
    print('Adding SRGz features to the nnmag counterparts...', '\n')

    class_df = pd.read_pickle(matched_class_path, compression='gzip')

    full_srgz_df = pd.read_pickle(srgz_cat_path, compression='gzip')
    full_srgz_df.srcname_fin = full_srgz_df.srcname_fin.str.decode('utf-8')

    # mutables are hard to merge, so here I make tuples form pdz arrays
    full_srgz_df['srgz_z_pdf'] = full_srgz_df['srgz_z_pdf'].fillna('').apply(tuple)

    # SRGz catalog (only best counterparts)
    srgz_df = full_srgz_df.query('srg_match_flag==1')

    print(f'Number of the SRGz sources: {len(srgz_df)}')
    print(f'Number of the nway sources: {len(class_df)}')

    zph_target_columns = [
        'ls_ra', 'ls_dec', 'srgz_z_max',
        'srgz_z_maxConf', 'srgz_z_merr68', 'srgz_z_perr68',
        'srgz_z_warning', 'srgz_z_pdf'
    ]

    srgz_slim_df = full_srgz_df[zph_target_columns].rename(
        columns={'srg_match_SQG': 'srgz_match_SQG'}
        )

    # Cross-match between nnmag optical counterparts and SRGz optical
    # counter. FULL SRGz catalog is used to get SRGz features for
    # all the nnmag optical cources.
    class_zph_df = lhf.cross_match_data_frames(
        class_df,
        srgz_slim_df,
        colname_ra1='desi_ra',
        colname_dec1='desi_dec',
        colname_ra2='ls_ra',
        colname_dec2='ls_dec',
        match_radius=1,
        df_prefix='nn',
        closest=True
    )

    slim_srgz_cols = [
        'srcname_fin', 'RA_fin', 'DEC_fin', 'ls_ra', 'ls_dec', 'g',
        'srgT_match_p0', 'srg_match_p', 'srg_match_pi', 'srg_match_warning',
        'srg_match_pstar', 'srg_match_pqso', 'srg_match_pgal', 'srg_match_SQG',
        'srgz_z_max', 'srgz_z_maxConf', 'srgz_z_pdf', 'srgz_z_merr68',
        'srgz_z_perr68', 'srgz_z_warning'
    ]

    cols2rename = {
        'srcname_fin': 'srgz_srcname_fin', 'RA_fin': 'srgz_RA_fin',
        'DEC_fin': 'srgz_DEC_fin', 'ls_ra': 'srgz_ls_ra',
        'ls_dec': 'srgz_ls_dec', 'g': 'srgz_g',
        'srgT_match_p0': 'srgz_match_p0', 'srg_match_p': 'srgz_match_p',
        'srg_match_pi': 'srgz_match_pi', 'srg_match_warning': 'srgz_match_warning',
        'srg_match_pstar': 'srgz_match_pstar', 'srg_match_pqso': 'srgz_match_pqso',
        'srg_match_pgal': 'srgz_match_pgal', 'srg_match_SQG': 'srgz_match_SQG'
    }

    columns2drop = [
        'srgz_srcname_fin', 'srgz_RA_fin', 'srgz_DEC_fin',
        'nn_ls_ra', 'nn_ls_dec', 'nn_sep',
        'nn_n_near', 'nn_n_matches'
    ]

    # Merge the updated nnmag catalog (containts SRGz features for every
    # nnmag counterpart) with the SRGz catalog using X-ray coordinates.
    # Result: evry nnmag counterpart has SRGz features AND every X-ray source
    # has two version of optical counterparts (nnmag and SRGz). Mostly these
    # two counterparts is the same optical source, but not all of them.
    # See the 'nn_srgz_same' column for details.

    # The SRGz counterparts has no spectral classification and need to be
    # cross-mathched with spectral catalogs using srgz_preprocess.py
    print()
    print('Adding SRGz counterparts to the X-ray sources...', '\n')
    class_zph_srgz_df = (class_zph_df.merge(
        srgz_df[slim_srgz_cols].rename(columns=cols2rename),
        left_on=['RA', 'DEC'],
        right_on=['srgz_RA_fin', 'srgz_DEC_fin'],
        suffixes=['', 'srgz_'], how='left')  # type: ignore
        ).drop(columns=columns2drop)

    # get rid of minus sign in srgz_z_merr68
    class_zph_srgz_df['srgz_z_merr68'] = -class_zph_srgz_df['srgz_z_merr68']

    # makrk close nnmag and srgz counterparts as the same
    nn_coords = SkyCoord(
        ra=class_zph_srgz_df['desi_ra'],
        dec=class_zph_srgz_df['desi_dec'],
        unit='deg'
        )
    srgz_coords = SkyCoord(
        ra=class_zph_srgz_df['srgz_ls_ra'],
        dec=class_zph_srgz_df['srgz_ls_dec'],
        unit='deg'
        )

    sep_nn_srgz = nn_coords.separation(srgz_coords).arcsec
    # class_zph_srgz_df['sep_nn_srgz'] = sep_nn_srgz

    nn_srgz_same = pd.Series(sep_nn_srgz) < 1
    class_zph_srgz_df.insert(31, 'nn_srgz_same', nn_srgz_same)

    # get rid of minus sign in nn_srgz_z_merr68
    class_zph_srgz_df['nn_srgz_z_merr68'] = -class_zph_srgz_df['nn_srgz_z_merr68']

    # create saving directory if it doesn't exist

    class_zph_srgz_df.to_pickle(srgz_prepared_path, compression='gzip')
    print(f'Preprocessed SRGz catalog is saved: {srgz_prepared_path}')


if __name__ == '__main__':
    main()
