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
from astropy.coordinates import SkyCoord


DATA_PATH = '/Users/mike/Repos/classification_LH/data'
SRGZ_PATH = '/Users/mike/Repos/XLF_LH/data/lhpv_03_23_sd01_a15_g14_srgz_CatA_XnX_model4_SQG_model5_v20221207'
SAVEPATH = 'data/output_data/srgz_nnmag.gz_pkl'
save_directory = os.path.dirname(SAVEPATH)


def main():

    print('\n', 'Adding SRGz features to the nnmag counterparts...', '\n')

    class_df = pd.read_pickle(
        'data/output_data/matched_and_classified.gz_pkl',
        compression='gzip'
        )

    full_srgz_df = pd.read_pickle(SRGZ_PATH, compression='gzip')
    full_srgz_df.srcname_fin = full_srgz_df.srcname_fin.str.decode('utf-8')

    # mutables are hard to merge, so here I make tuples form pdz arrays
    full_srgz_df['srgz_z_pdf'] = full_srgz_df['srgz_z_pdf'].fillna('').apply(tuple)

    # SRGz catalog (only best counterparts)
    srgz_df = full_srgz_df.query('srg_match_flag==1')

    print(f'Всего источников SRGz: {len(srgz_df)}')
    print(f'Всего источников nway: {len(class_df)}')

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

    # class_zph_df.to_pickle(f'{save_directory}/class_zph.pkl')

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
    print('\n', 'Adding SRGz counterparts to the X-ray sources...', '\n')
    class_zph_srgz_df = (class_zph_df.merge(
        srgz_df[slim_srgz_cols].rename(columns=cols2rename),
        left_on=['RA', 'DEC'],
        right_on=['srgz_RA_fin', 'srgz_DEC_fin'],
        suffixes=['', 'srgz_'], how='left')
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
    class_zph_srgz_df['sep_nn_srgz'] = sep_nn_srgz

    nn_srgz_same = pd.Series(sep_nn_srgz) < 1
    class_zph_srgz_df.insert(31, 'nn_srgz_same', nn_srgz_same)

    # get rid of minus sign in nn_srgz_z_merr68
    class_zph_srgz_df['nn_srgz_z_merr68'] = -class_zph_srgz_df['nn_srgz_z_merr68']

    # create saving directory if it doesn't exist

    Path(save_directory).mkdir(parents=True, exist_ok=True)

    class_zph_srgz_df.to_pickle(SAVEPATH, compression='gzip')
    print(f'Preprocessed SRGz catalog is saved: {SAVEPATH}')


if __name__ == '__main__':
    main()
