#!/usr/bin/env python3

import re
import pandas as pd
import numpy as np
from lh_class import lh_functions as lhf
from uncertainties import unumpy
from functools import reduce
from pathlib import Path
import argparse
pd.options.mode.chained_assignment = None

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

# DESI soruces in LH
# https://www.notion.so/LH-data-95f7ad4a14cc4b2d8ef4e3a3237bd29b?pvs=4#a0fa7d64a06a42f4b8ed8a03d25cb736
DESI_PATH = input_dir_path + 'desi_mask_lh.gz_pkl'

# GAIA sources in LH
# https://www.notion.so/LH-data-95f7ad4a14cc4b2d8ef4e3a3237bd29b?pvs=4#781138565e554d4e935a1ce5651db3ca
GAIA_PATH = input_dir_path + 'gaia_dr3_astroph_4-result.csv'

# SDSS sources in LH
# https://www.notion.so/LH-data-95f7ad4a14cc4b2d8ef4e3a3237bd29b?pvs=4#2f3fedf2068746949dc2cddbdb201a90
SDSS_PATH = input_dir_path + 'sdss_tap.csv'

# DESI positional errors, downloaded separately to merge with SB catalogue
# The reason they are here is to avoid re-running the SB piplene
# DESI_ERR_PATH = f'{DATA_BASE_PATH}/desi/desi_lh_coord_errors.gz_pkl'

# SB nnmag counterparts
# https://www.notion.so/LH-data-95f7ad4a14cc4b2d8ef4e3a3237bd29b?pvs=4#e61e989c1773400aae34c6f984012a2e
# TODO: check if the cross-match file is up to date (why some date is in the name?)
NNMAG_CAT_FILENAME = 'ERO_lhpv_03_23_sd01_a15_g14_desi_nway_match_21_10_22.gz_pkl'
DESI_MATCH_PATH = input_dir_path + NNMAG_CAT_FILENAME
MILQ_PATH = input_dir_path + 'milliquas_LH.csv'
SIMBAD_PATH = input_dir_path + 'simbad_df.pkl'

# path to save the table of sources matched with external catalogs
# and classified as extragalactic/not extragalactic
# this is mediate (auxiliary) data
mediate_dir_path = args.output_path + 'mediate_data/'
# create saving directory if it doesn't exist
Path(mediate_dir_path).mkdir(parents=True, exist_ok=True)

# path to the file produced by the srgz_preprocess.py
srgz_prep_path = mediate_dir_path + 'srgz_nnmag.gz_pkl'

# path to the srgz_spec file
srgz_spec_path = mediate_dir_path + 'srgz_nnmag_spec.gz_pkl'

# path to the matched and classified catalog
matched_class_path = mediate_dir_path + 'matched_and_classified.gz_pkl'

# path to the result catalog
result_dir_path = args.output_path + 'result_data/'
# create saving directory if it doesn't exist
Path(result_dir_path).mkdir(parents=True, exist_ok=True)
result_path = result_dir_path + 'lh_nnmag_srgz.gz_pkl'

# ECF given by MG in 2022
ECF_MG_241122 = 0.7228


def spectral_parsing(mode='nnmag'):

    # cross-match with spectral catalogs and databases

    print('Reading the input catalogs...', '\n')
    desi_df = pd.read_pickle(DESI_PATH, compression='gzip')
    print('DESI sources in LH:', len(desi_df))

    gaia_df = pd.read_csv(GAIA_PATH)
    print('GAIA sources in LH:', len(gaia_df))

    # TODO: convert Simbad parsing to a script
    simbad_df = pd.read_pickle(SIMBAD_PATH)
    print('SIMBAD sources in LH:', len(simbad_df), '\n')

    if mode == 'nnmag':

        # coordinates of sources which needs spectral classification
        DESI_RA_NAME = 'desi_ra'
        DESI_DEC_NAME = 'desi_dec'

        XRAY_RA_NAME = 'RA_fin'
        XRAY_DEC_NAME = 'DEC_fin'

        SRC_NAME = 'srcname_fin'

        # read nnmag match results
        catalog_to_classify = pd.read_pickle(DESI_MATCH_PATH, compression='gzip')
        # TODO: up-to-date nnmag catalog needs no index reset
        catalog_to_classify.reset_index(drop=True, inplace=True)
        print('DESI nnmag matches:', len(catalog_to_classify))

        # TODO: update fluxes in accordance with the future agreements
        catalog_to_classify['flux_05-20_LH'] = catalog_to_classify['ML_FLUX_0'] * ECF_MG_241122
        catalog_to_classify['flux_05-20_LH_ERR'] = catalog_to_classify['ML_FLUX_ERR_0'] * ECF_MG_241122

    elif mode == 'srgz':

        # coordinates of sources which needs spectral classification
        DESI_RA_NAME = 'srgz_ls_ra'
        DESI_DEC_NAME = 'srgz_ls_dec'

        XRAY_RA_NAME = 'RA'
        XRAY_DEC_NAME = 'DEC'

        SRC_NAME = 'NAME'

        # read srgz_preprocess.py result
        catalog_to_classify = pd.read_pickle(srgz_prep_path, compression='gzip')

        # drop several sources with problem coordinates
        catalog_to_classify = catalog_to_classify.dropna(
            subset=[DESI_RA_NAME, DESI_DEC_NAME]
            )

    else:
        raise ValueError('`mode` value is incorrect. Try using "nnmag" or "srgz".')

    merge_on_columns = list(catalog_to_classify.columns.values)

    print('* ' * 15)
    print('GAIA PREROCESSING...', '\n')

    # total porper motion and its error
    uncert_pmra = unumpy.uarray(gaia_df['pmra'], gaia_df['pmra_error'])
    uncert_pmdec = unumpy.uarray(gaia_df['pmdec'], gaia_df['pmdec_error'])
    upm = (uncert_pmra ** 2 + uncert_pmdec ** 2) ** .5
    gaia_df['pm'] = unumpy.nominal_values(upm)
    gaia_df['pm_error'] = unumpy.std_devs(upm)

    print('GAIA CROSS-MATCH WITH ERO NNMAG...', '\n')

    desi_gaia_df = lhf.cross_match_data_frames(
        df1=catalog_to_classify,
        df2=gaia_df,
        colname_ra1=DESI_RA_NAME,
        colname_dec1=DESI_DEC_NAME,
        colname_ra2='ra',
        colname_dec2='dec',
        match_radius=.5,
        df_prefix='GAIA',
        closest=True,
        solo_near=True,
        ero_sep=True,
        xray_ra=XRAY_RA_NAME,
        xray_dec=XRAY_DEC_NAME
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
        df1=catalog_to_classify,
        df2=sdss_spectral,
        colname_ra1=DESI_RA_NAME,
        colname_dec1=DESI_DEC_NAME,
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
        df1=catalog_to_classify,
        df2=milq_df,
        colname_ra1=DESI_RA_NAME,
        colname_dec1=DESI_DEC_NAME,
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

    # leave only the closest neighbours
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

    # these strings contain the word 'Galaxy' but appear to be AGN
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

    simbad_closest_df = simbad_closest_df[simbad_useful_columns]

    print('JOINING ALL MATCHES...', '\n')
    # merge desi_gaia with desi_sdss with desi to create a new desi_gaia_sdss catalogue
    # https://stackoverflow.com/questions/44327999/python-pandas-merge-multiple-dataframes
    data_frames = [catalog_to_classify, desi_gaia_df, desi_sdss_df, desi_milq_df]

    all_matched_df = reduce(
        lambda left, right: pd.merge(
            left, right, on=merge_on_columns, how='outer'), data_frames
            ).fillna(np.nan)

    all_matched_df.fillna(np.nan, inplace=True)

    # add closest Simbad counterparts
    # TODO: left_on and right_on column names should be the same
    all_matched_df = all_matched_df.merge(
        simbad_closest_df,
        left_on=SRC_NAME,
        right_on='srcname_fin',
        how='left'
        )

    # classification: spectral type
    print('PRELIMINARY CLASSIFICATION...', '\n')
    print('Joining subclasses in bigger groups for every external input catalog...', '\n')
    # add flags for GAIA stars
    spec_class_df = lhf.star_marker(all_matched_df, s_n_threshold=5)

    print('GAIA stat:')
    print(spec_class_df.class_GAIA_class.value_counts().to_string(), '\n')

    spec_class_df.loc[:, 'class_SDSS_class'] = spec_class_df.SDSS_spCl.str.strip()

    # joining SDSS subclasses
    # SDSS_class GALAXY with SDSS_subCl AGN is QSO
    spec_class_df['class_SDSS_class'] = (
        spec_class_df['class_SDSS_class']
        .where(spec_class_df['SDSS_subCl'] != "AGN", 'QSO')
        )

    print('SDSS stat:')
    print(spec_class_df['class_SDSS_class'].value_counts().to_string(), '\n')

    # joining MILQ subclasses
    milq_qso_types = ['QSO', 'AGN', 'AGN-NL', 'QSO-NL', 'BLAZAR', 'RADIO-SOURCE']
    mask_extragal = spec_class_df.eval("MILQ_TYPE_spec in @milq_qso_types")
    spec_class_df.loc[mask_extragal, 'class_MILQ_class'] = 'QSO'

    mask_extragal = spec_class_df.eval("MILQ_TYPE_spec in ['GALAXY']")
    spec_class_df.loc[mask_extragal, 'class_MILQ_class'] = 'GALAXY'

    mask_extragal = spec_class_df.eval("MILQ_TYPE_spec in ['STAR', 'MOVING-OBJECT']")
    spec_class_df.loc[mask_extragal, 'class_MILQ_class'] = 'STAR'

    print('MILQ stat:')
    print(spec_class_df['class_MILQ_class'].value_counts().to_string(), '\n')

    spec_class_df['desi_extended'] = spec_class_df['desi_type'] != 'PSF'

    spec_class_df.loc[:, 'class_final'] = spec_class_df.apply(
        lhf.assign_final_class, axis=1
        )
    spec_class_df.loc[:, 'class_source'] = spec_class_df.apply(
        lhf.assign_class_source, axis=1
        )

    spec_class_df.loc[:, 'class_source_index'] = spec_class_df.apply(
        lhf.assign_class_source_index, axis=1
        )

    not_gaia_star_mask = (
        (spec_class_df['class_GAIA_class'] != 'GALACTIC') &
        (spec_class_df['class_final'] == 'GALACTIC (non-Gaia)')
        )

    spec_class_df['class_final'] = np.where(
        not_gaia_star_mask, 'GALACTIC (non-Gaia)', spec_class_df['class_final']
        )

    spec_class_df.loc[:, 'redshift_final'] = spec_class_df.apply(
        lhf.assign_final_redshift, axis=1
        )
    spec_class_df.loc[:, 'z_spec_origin'] = spec_class_df.apply(
        lhf.assign_redshift_source, axis=1
        )

    # order of appearance in the table (for the scatterplot)
    zorder_dict = {
        'GALAXY': 1,
        'GALACTIC': 2,
        'GALACTIC (non-Gaia)': 3,
        'QSO': 4,
        'STAR': 0,
        'milq moving': 6,
        'UNKNOWN': 5
        }

    spec_class_df['zorder'] = spec_class_df['class_final'].replace(zorder_dict)
    spec_class_df = spec_class_df.sort_values(by='zorder', ascending=False)

    final_class_stat = pd.DataFrame(spec_class_df.class_final.value_counts())
    print()
    print(final_class_stat.to_string(), '\n')

    if mode == 'nnmag':
        return spec_class_df

    elif mode == 'srgz':

        # TODO: make renaming process less messy
        columns2rename = {
            'srgz_spec_NAME': 'srcname',
            'srgz_spec_class_source': 'srgz_spec_class_origin',
            'srgz_spec_class_source_index': 'srgz_spec_class_origin_id',
            'srgz_spec_class_final': 'srgz_spec_class',
            'srgz_spec_redshift_final': 'srgz_spec_z'
        }

        useful_columns = [
            'NAME', 'class_final', 'redshift_final',
            'class_source', 'class_source_index',
        ]

        (
            all_matched_df[useful_columns]
            .add_prefix('srgz_spec_')
            .rename(columns=columns2rename)
            .reset_index(drop=True)
            .to_pickle(srgz_spec_path, compression='gzip')
        )

        print(f'Catalog of SRGz sources with spectral classes is saved: {srgz_spec_path}', '\n')


def main():

    spec_class_df = spectral_parsing(mode='nnmag')

    # count sources without classification
    filtered_spec_class_df = spec_class_df.query(
        '~(desi_rel_dered_mag_z.isna() & class_final=="UNKNOWN")'
        )
    print()
    print('Total:', len(spec_class_df))
    print('Without missed values:', len(filtered_spec_class_df))

    data_loss = len(spec_class_df) - len(filtered_spec_class_df)
    data_loss_percent = 1 - len(filtered_spec_class_df) / len(spec_class_df)
    print(f'Data loss due to missed values: {data_loss_percent:.1%} ({data_loss})', '\n')

    print('FINAL CLASSIFICATION...', '\n')
    filtered_spec_class_df['extragal'] = filtered_spec_class_df.apply(
        lhf.extragal_classifier, axis=1
        )
    print(filtered_spec_class_df['extragal'].value_counts().to_string(), '\n')

    # add final classification
    spec_class_df['is_extragal'] = spec_class_df.apply(lhf.extragal_classifier, axis=1)

    paper_columns = [
        'srcname_fin', 'RA_fin', 'DEC_fin', 'pos_r98', 'DET_LIKE_0',
        'ML_CTS_0', 'ML_CTS_ERR_0', 'ML_RATE_0', 'ML_RATE_ERR_0', 'ML_BKG_0',
        'ML_EXP_1', 'flux_05-20_LH', 'flux_05-20_LH_ERR', 'DIST_NN',
        'NH', 'desi_id', 'desi_ra', 'desi_dec', 'desi_dered_mag_g',
        'desi_dered_mag_r', 'desi_dered_mag_z', 'desi_dered_mag_w1',
        'desi_dered_mag_w2', 'desi_type', 'nway_prob_has_match',
        'nway_prob_this_match', 'class_final', 'redshift_final',
        'z_spec_origin', 'class_source', 'class_source_index', 'is_extragal'
    ]

    columns_to_rename = {
        'srcname_fin': 'NAME', 'RA_fin': 'RA', 'DEC_fin': 'DEC',
        'pos_r98': 'POS_R98', 'DET_LIKE_0': 'DET_LIKE', 'ML_CTS_0': 'CTS',
        'ML_CTS_ERR_0': 'CTS_ERR', 'ML_RATE_0': 'SRC_RATE',
        'ML_RATE_ERR_0': 'SRC_RATE_ERR', 'ML_BKG_0': 'BKG_RATE',
        'ML_EXP_1': 'EXP', 'DIST_NN': 'DIST_CN', 'flux_05-20_LH': 'FLUX_05-20',
        'flux_05-20_LH_ERR': 'FLUX_05-20_ERR', 'class_source': 'class_origin',
        'class_source_index': 'class_origin_index', 'class_final': 'src_class',
        'redshift_final': 'redshift', 'nway_prob_has_match': 'desi_p_any',
        'nway_prob_this_match': 'desi_p_i'
        }

    paper_cat_df = (
        spec_class_df[paper_columns]
        .rename(columns=columns_to_rename)
        .reset_index(drop=True)
        )

    # save the matched and classified catalog
    paper_cat_df.to_pickle(matched_class_path, compression='gzip')

    print(f'Merged and classified catalog is saved: {matched_class_path}')


def srgz_spec():
    spectral_parsing(mode='srgz')


def postprocess():

    print('Reading the input catalogs...', '\n')
    class_zph_srgz_df = pd.read_pickle(
        srgz_prep_path,
        compression='gzip'
        )

    srgz_spec_df = pd.read_pickle(srgz_spec_path, compression='gzip')

    print('Merging srgz_nnmag with srgz_nnmag_spec...', '\n')
    class_zph_srgz_spec_df = class_zph_srgz_df.merge(
        srgz_spec_df,
        right_on='srcname',
        how='left',
        left_on='NAME'
        ).drop(columns=['srcname'])

    print('Columns renaming and reduction...', '\n')
    for column in class_zph_srgz_spec_df.columns:

        colname_elements = column.split('_')

        if '_z_' in column:
            new_root = column.replace('_z_', '_zph_')
            # print(column, '->', new_root, '\n')
            class_zph_srgz_spec_df.rename(columns={column: new_root}, inplace=True)

        elif colname_elements[0] == 'desi':
            class_zph_srgz_spec_df.rename(columns={column: f'nn_{column}'}, inplace=True)

    final_rename_dict = {
        'src_class': 'nn_spec_class',
        'redshift': 'nn_spec_z',
        'class_origin': 'nn_spec_class_origin',
        'class_origin_index': 'nn_spec_class_id',
        'is_extragal': 'nn_is_extragal',
        'srgz_spec_class_origin_id': 'srgz_spec_class_id',
        'srgz_zph_max': 'srgz_zph',
        'srgz_zph_maxConf': 'srgz_zconf',
        'nn_desi_p_any': 'nn_p_any',
        'nn_desi_p_i': 'nn_p_i',
        'nn_srgz_zph_max': 'nn_srgz_zph',
        'nn_srgz_zph_maxConf': 'nn_srgz_zconf',
        'nn_srgz_zph_merr68': 'nn_srgz_zph_nerr68',
        'srgz_zph_merr68': 'srgz_zph_nerr68',
        'z_spec_origin': 'nn_spec_z_origin',
    }

    class_zph_srgz_spec_df.rename(columns=final_rename_dict, inplace=True)

    # A new version of SRGz table contains a bunch of `object` dtypes columns.
    # I exclude them to replace `inf` with `nan` in other columns.
    not_obj_type_list = list(class_zph_srgz_spec_df.select_dtypes(exclude=['object']).columns)

    # get rid of infinities in not_obj_type_list
    class_zph_srgz_spec_df[not_obj_type_list] = (
        class_zph_srgz_spec_df[not_obj_type_list]
        .replace([np.inf, -np.inf], np.nan)
        )

    # drop nn_spec_z if origin of class and redshift are different
    mask = (
        ~class_zph_srgz_spec_df['nn_spec_z'].isna() &
        (class_zph_srgz_spec_df['nn_spec_class_origin'] !=
         class_zph_srgz_spec_df['nn_spec_z_origin'])
            )

    class_zph_srgz_spec_df['nn_spec_z'][mask] = np.nan

    class_zph_srgz_spec_df.drop(columns=['nn_spec_z_origin'], inplace=True)

    class_zph_srgz_spec_df.to_pickle(result_path, compression='gzip')
    print(f'Final catalog is saved: {result_path}')


if __name__ == '__main__':
    main()
