#!/usr/bin/env python3

import re
import pandas as pd
import numpy as np
import seaborn as sns
from lh_class import lh_functions as lhf
from uncertainties import unumpy
from functools import reduce
pd.options.mode.chained_assignment = None

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
MILQ_PATH = f'{DATA_BASE_PATH}/SB/milliquas_LH.csv'
SIMBAD_PATH = f'{DATA_BASE_PATH}/simbad_df.pkl'

# ECF given by MG in 2022
ECF_MG_241122 = 0.7228


def main():

    print()
    print('Welcome to the LH classification script!', '\n')

    print('Reading the input catalogs...')
    desi_df = pd.read_pickle(DESI_PATH, compression='gzip')
    print('DESI sources in LH:', len(desi_df))

    gaia_df = pd.read_csv(GAIA_PATH)
    print('GAIA sources in LH:', len(gaia_df))

    # read nnmag match results
    ero_desi_nnmag_df = pd.read_pickle(DESI_MATCH_PATH, compression='gzip')
    # TODO: up to date nnmag catalog needs no index reset
    ero_desi_nnmag_df.reset_index(drop=True, inplace=True)
    print('DESI nnmag matches:', len(ero_desi_nnmag_df))

    ero_desi_nnmag_df['flux_05-20_LH'] = ero_desi_nnmag_df['ML_FLUX_0'] * ECF_MG_241122
    ero_desi_nnmag_df['flux_05-20_LH_ERR'] = ero_desi_nnmag_df['ML_FLUX_ERR_0'] * ECF_MG_241122
    erosita_columns = list(ero_desi_nnmag_df.columns.values)

    # TODO: convert Simbad parsing to a script
    simbad_df = pd.read_pickle(SIMBAD_PATH)
    print('SIMBAD sources in LH:', len(simbad_df), '\n')

    print('* ' * 15)
    print('GAIA PREROCESSING...', '\n')

    # Total porper motion and its error
    uncert_pmra = unumpy.uarray(gaia_df['pmra'], gaia_df['pmra_error'])
    uncert_pmdec = unumpy.uarray(gaia_df['pmdec'], gaia_df['pmdec_error'])
    upm = (uncert_pmra ** 2 + uncert_pmdec ** 2) ** .5
    gaia_df['pm'] = unumpy.nominal_values(upm)
    gaia_df['pm_error'] = unumpy.std_devs(upm)

    print('GAIA CROSS-MATCH WITH ERO NNMAG...', '\n')

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

    print('* ' * 15)
    print('SDSS PREPROCESSING...', '\n')

    sdss_df = pd.read_csv(SDSS_PATH)
    sdss_spectral = sdss_df.query('~spCl.isna()')

    # radec error of SDSS sources
    sdss_radec_err_sec = np.sqrt(
        sdss_spectral['e_RA_ICRS']**2 + sdss_spectral['e_DE_ICRS']**2
        )
    sdss_spectral['radec_err_sec'] = sdss_radec_err_sec

    print('SDSS CROSS-MATCH WITH ERO NNMAG...', '\n')

    desi_sdss_df = lhf.cross_match_data_frames(
        df1=ero_desi_nnmag_df,
        df2=sdss_spectral,
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

    print(desi_sdss_df.SDSS_spCl.value_counts().to_string(), '\n')
    print('________')
    desi_sdss_stat = desi_sdss_df.groupby('SDSS_spCl')['SDSS_subCl'].value_counts()
    print(desi_sdss_stat.to_string(), '\n')

    print('* ' * 15)
    print('MILQ PREPROCESSING...', '\n')
    milq_df = pd.read_csv(MILQ_PATH, sep=',')
    # throwing out trailing whitespaces
    milq_df.NAME = milq_df.NAME.str.strip()

    # photometric redshifts are rounded to 1 decimal place in the catalog,
    # only spectral redshifts and types are kept
    milq_df['Z_spec'] = milq_df['Z'][~(milq_df['Z'].round(1) == milq_df['Z'])]
    milq_df['TYPE_spec'] = milq_df['TYPE'][~(milq_df['Z'].round(1) == milq_df['Z'])]

    print('MILQ CROSS-MATCH WITH ERO NNMAG...', '\n')
    desi_milq_df = lhf.cross_match_data_frames(
        df1=ero_desi_nnmag_df,
        df2=milq_df,
        colname_ra1='desi_ra',
        colname_dec1='desi_dec',
        colname_ra2='RA',
        colname_dec2='DEC',
        match_radius=1,
        df_prefix='MILQ',
        closest=True
    )

    print('MILQ POSTROCESSING...', '\n')

    # throwing out trailing whitespaces
    desi_milq_df.MILQ_TYPE_spec = desi_milq_df.MILQ_TYPE_spec.str.strip()

    # remove R (radio), X(X-ray), 2(lobes) letters from the classification
    desi_milq_df['MILQ_TYPE_spec'] = desi_milq_df.MILQ_TYPE_spec.str.replace('R', '')
    desi_milq_df['MILQ_TYPE_spec'] = desi_milq_df.MILQ_TYPE_spec.str.replace('X', '')
    desi_milq_df['MILQ_TYPE_spec'] = desi_milq_df.MILQ_TYPE_spec.str.replace('2', '')

    # ease the classification
    type_dict_milq = {
        'Q': 'QSO',
        'A': 'AGN',
        'B': 'BLAZAR',
        'L': 'QSO-LENSED',
        'K': 'QSO-NL',
        'N': 'AGN-NL',
        'G': 'GALAXY',
        'S': 'STAR',
        'C': 'CV',
        'M': 'MOVING-OBJECT',
        '': 'RADIO-SOURCE',
    }

    desi_milq_df['MILQ_TYPE_spec'] = desi_milq_df.MILQ_TYPE_spec.map(type_dict_milq)

    print(desi_milq_df['MILQ_TYPE_spec'].value_counts().to_string(), '\n')

    print('SIMBAD PREPROCESSING...')
    # leave only reliable z values
    simbad_df['z_rel_mask'] = (
        (simbad_df['RVZ_TYPE_z'] == 'z') &
        simbad_df['RVZ_QUAL'].isin(["A", "B", "C", "D"])
        )

    simbad_df['SIMBAD_z_rel'] = np.where(
        simbad_df['z_rel_mask'], simbad_df['Z_VALUE'], np.nan
        )

    print('JOINING ALL MATCHES...', '\n')
    # merge desi_gaia with desi_sdss with desi to create a new desi_gaia_sdss catalogue
    # https://stackoverflow.com/questions/44327999/python-pandas-merge-multiple-dataframes
    data_frames = [ero_desi_nnmag_df, desi_gaia_df, desi_sdss_df, desi_milq_df]

    all_matched_df = reduce(
        lambda left, right: pd.merge(
            left, right, on=erosita_columns, how='outer'), data_frames
            ).fillna(np.nan)

    all_matched_df.fillna(np.nan, inplace=True)

    print('PRELIMINARY CLASSIFICATION...', '\n')
    print('Joining subclasses in bigger groups for every external input catalog', '\n')
    # add flags for GAIA stars
    all_matched_df = lhf.star_marker(all_matched_df, s_n_threshold=5)

    print('GAIA stat:')
    print(all_matched_df.class_GAIA_class.value_counts().to_string(), '\n')

    all_matched_df.loc[:, 'class_SDSS_class'] = all_matched_df.SDSS_spCl.str.strip()

    # joining SDSS subclasses
    # SDSS_class GALAXY with SDSS_subCl AGN is QSO
    all_matched_df['class_SDSS_class'] = (
        all_matched_df['class_SDSS_class']
        .where(all_matched_df['SDSS_subCl'] != "AGN", 'QSO')
        )

    print('SDSS stat:')
    print(all_matched_df['class_SDSS_class'].value_counts().to_string(), '\n')

    # joining MILQ subclasses
    milq_qso_types = ['QSO', 'AGN', 'AGN-NL', 'QSO-NL', 'BLAZAR', 'RADIO-SOURCE']
    mask_extragal = all_matched_df.eval("MILQ_TYPE_spec in @milq_qso_types")
    all_matched_df.loc[mask_extragal, 'class_MILQ_class'] = 'QSO'

    mask_extragal = all_matched_df.eval("MILQ_TYPE_spec in ['GALAXY']")
    all_matched_df.loc[mask_extragal, 'class_MILQ_class'] = 'GALAXY'

    mask_extragal = all_matched_df.eval("MILQ_TYPE_spec in ['STAR', 'MOVING-OBJECT']")
    all_matched_df.loc[mask_extragal, 'class_MILQ_class'] = 'STAR'

    print('MILQ stat:')
    print(all_matched_df['class_MILQ_class'].value_counts().to_string(), '\n')

    simbad_closest_df = simbad_df.loc[
        simbad_df.groupby('srcname_fin').DISTANCE_RESULT.idxmin()
        ]

    # rough classification to better visualize the data
    # TODO: move this task to a deducated Simbad script
    def find_whole_word(w):
        return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

    rough_class_dict = {}

    for obj_type in list(simbad_closest_df['OTYPE_V'].unique()):
        if bool(find_whole_word('galaxy')(obj_type)):
            rough_class_dict[obj_type] = 'GALAXY'
        elif bool(find_whole_word('star')(obj_type)):
            rough_class_dict[obj_type] = 'GALACTIC (non-Gaia)'
        elif bool(find_whole_word('quasar')(obj_type)):
            rough_class_dict[obj_type] = 'QSO'
        else:
            rough_class_dict[obj_type] = 'UNKNOWN'

    simbad_closest_df['OTYPE_rough'] = (
        simbad_closest_df['OTYPE_V'].replace(rough_class_dict)
        )

    # These strings contain word 'Galaxy' but appear to be AGN
    additional_qso = [
        'Active Galaxy Nucleus',
        'Seyfert 1 Galaxy',
        'Seyfert 2 Galaxy',
        'LINER-type Active Galaxy Nucleus',
        'Blazar'
        ]

    simbad_add_qso = (
        simbad_closest_df['OTYPE_V']
        .isin(additional_qso)
        .replace({True: 'QSO', False: np.nan})
        )

    simbad_add_qso = simbad_add_qso.fillna(simbad_closest_df['OTYPE_rough'])
    simbad_closest_df['class_SIMBAD_class'] = simbad_add_qso

    simbad_closest_df['class_SIMBAD_class'].value_counts()

    simbad_useful_columns = [
        'srcname_fin', 'MAIN_ID', 'class_SIMBAD_class',
        'OTYPE_V', 'SIMBAD_z_rel', 'RVZ_ERROR'
        ]

    class_df = all_matched_df.merge(
        simbad_closest_df[simbad_useful_columns],
        on='srcname_fin', how='left'
        )

    class_df['desi_extended'] = class_df['desi_type'] != 'PSF'

    class_df.loc[:, 'class_final'] = class_df.apply(
        lhf.assign_final_class, axis=1
        )
    class_df.loc[:, 'class_source'] = class_df.apply(
        lhf.assign_class_source, axis=1
        )
        
    class_df.loc[:, 'class_source_index'] = class_df.apply(
        lhf.assign_class_source_index, axis=1
        )

    not_gaia_star_mask = (
        (class_df['class_GAIA_class'] != 'GALACTIC') &
        (class_df['class_final'] == 'GALACTIC (non-Gaia)')
        )

    class_df['class_final'] = np.where(
        not_gaia_star_mask, 'GALACTIC (non-Gaia)', class_df['class_final']
        )

    class_df.loc[:, 'redshift_final'] = class_df.apply(
        lhf.assign_final_redshift, axis=1
        )
    class_df.loc[:, 'z_spec_origin'] = class_df.apply(
        lhf.assign_redshift_source, axis=1
        )

    # Order of appearance in the table (and scatterplot)
    zorder_dict = {
        'GALAXY': 1,
        'GALACTIC': 2,
        'GALACTIC (non-Gaia)': 3,
        'QSO': 4,
        'STAR': 0,
        'milq moving': 6,
        'UNKNOWN': 5
        }

    class_df['zorder'] = class_df['class_final'].replace(zorder_dict)
    class_df = class_df.sort_values(by='zorder', ascending=False)

    final_class_stat = pd.DataFrame(class_df.class_final.value_counts())
    print(final_class_stat.to_string(), '\n')

    filtered_class_df = class_df.query(
        '~(desi_rel_dered_mag_z.isna() & class_final=="UNKNOWN")'
        )
    print()
    print('Всего:', len(class_df))
    print('Без пропусков:', len(filtered_class_df))

    data_loss = len(class_df) - len(filtered_class_df)
    data_loss_percent = 1 - len(filtered_class_df) / len(class_df)
    print(f'Потеряно из-за пропусков: {data_loss_percent:.1%} ({data_loss})')

    print('FINAL CLASSIFICATION...')
    filtered_class_df['extragal'] = filtered_class_df.apply(
        lhf.extragal_classifier, axis=1
        )
    print(filtered_class_df['extragal'].value_counts().to_string())


if __name__ == '__main__':
    main()
