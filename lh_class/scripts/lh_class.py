#!/usr/bin/env python3

import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
import seaborn as sns
from lh_class import lh_functions as lhf
from uncertainties import unumpy

DATA_BASE_PATH = '/Users/mike/Repos/classification_LH/data'
# DESI soruces in LH
# https://www.notion.so/LH-data-95f7ad4a14cc4b2d8ef4e3a3237bd29b?pvs=4#a0fa7d64a06a42f4b8ed8a03d25cb736
DESI_PATH = f'{DATA_BASE_PATH}/desi/desi_mask_lh.gz_pkl'

# GAIA sources in LH
# https://www.notion.so/LH-data-95f7ad4a14cc4b2d8ef4e3a3237bd29b?pvs=4#781138565e554d4e935a1ce5651db3ca
GAIA_PATH = f'{DATA_BASE_PATH}/gaia/gaia_dr3_astroph_4-result.csv'

# SDSS sources in LH
# https://www.notion.so/LH-data-95f7ad4a14cc4b2d8ef4e3a3237bd29b?pvs=4#2f3fedf2068746949dc2cddbdb201a90
SDSS_PATH = f'{DATA_BASE_PATH}/SDSS/sdss_tap.csv'

# DESI positional errors, downloaded separately to merge with SB catalogue
# The reason they are here is to avoid re-running the SB piplene
# DESI_ERR_PATH = f'{DATA_BASE_PATH}/desi/desi_lh_coord_errors.gz_pkl'

# SB nnmag counterparts
# https://www.notion.so/LH-data-95f7ad4a14cc4b2d8ef4e3a3237bd29b?pvs=4#e61e989c1773400aae34c6f984012a2e
# TODO: check if the cross-match file is up to date (why some date is in the name?)
NNMAG_CAT_FILENAME = 'ERO_lhpv_03_23_sd01_a15_g14_desi_nway_match_21_10_22.gz_pkl'
DESI_MATCH_PATH = f'{DATA_BASE_PATH}/SB/{NNMAG_CAT_FILENAME}'

ECF_MG_241122 = 0.7228


def main():

    print()
    print('Welcome to the LH classification script!', '\n')

    print('Reading the input data...')
    desi_df = pd.read_pickle(DESI_PATH, compression='gzip')
    print('DESI sources in LH:', len(desi_df))

    gaia_df = pd.read_csv(GAIA_PATH)
    print('GAIA sources in LH:', len(gaia_df))

    # read nnmag match results
    ero_desi_nnmag_df = pd.read_pickle(DESI_MATCH_PATH, compression='gzip')
    # TODO: up to date nnmag catalog needs no index reset
    ero_desi_nnmag_df.reset_index(drop=True, inplace=True)
    print('DESI nnmag matches:', len(ero_desi_nnmag_df), '\n')

    ero_desi_nnmag_df['flux_05-20_LH'] = ero_desi_nnmag_df['ML_FLUX_0'] * ECF_MG_241122
    ero_desi_nnmag_df['flux_05-20_LH_ERR'] = ero_desi_nnmag_df['ML_FLUX_ERR_0'] * ECF_MG_241122
    erosita_columns = list(ero_desi_nnmag_df.columns.values)

    print('* ' * 15)
    print('GAIA CROSS-MATCH WITH ERO NNMAG', '\n')

    # Total porper motion and its error
    uncert_pmra = unumpy.uarray(gaia_df['pmra'], gaia_df['pmra_error'])
    uncert_pmdec = unumpy.uarray(gaia_df['pmdec'], gaia_df['pmdec_error'])
    upm = (uncert_pmra ** 2 + uncert_pmdec ** 2) ** .5
    gaia_df['pm'] = unumpy.nominal_values(upm)
    gaia_df['pm_error'] = unumpy.std_devs(upm)

    desi_gaia_df = lhf.cross_match_data_frames(
        df1=ero_desi_nnmag_df,
        df2=gaia_df,
        colname_ra1='desi_ra',
        colname_dec1='desi_dec',
        colname_ra2='ra',
        colname_dec2='dec',
        match_radius=.5,
        df_prefix='GAIA',
        closest=True,
        solo_near=True,
        ero_sep=True
    )

    print('* ' * 10)
    print('SDSS CROSS-MATCH WITH ERO NNMAG', '\n')

    sdss_df = pd.read_csv(SDSS_PATH)
    sdss_spectral = sdss_df.query('~spCl.isna()')

    # radec error of SDSS sources
    sdss_radec_err_sec = np.sqrt(
        sdss_spectral['e_RA_ICRS']**2 + sdss_spectral['e_DE_ICRS']**2
        )
    sdss_spectral['radec_err_sec'] = sdss_radec_err_sec

    desi_sdss_df = lhf.cross_match_data_frames(
        ero_desi_nnmag_df,
        sdss_spectral,
        colname_ra1='desi_ra',
        colname_dec1='desi_dec',
        colname_ra2='RA_ICRS',
        colname_dec2='DE_ICRS',
        match_radius=1,
        df_prefix='SDSS',
        closest=True
    )

    desi_sdss_df['SDSS_NAME'] = desi_sdss_df['SDSS_SDSS16']
    desi_sdss_df['SDSS_subCl'] = desi_sdss_df['SDSS_subCl'].str.strip()


if __name__ == '__main__':
    main()
