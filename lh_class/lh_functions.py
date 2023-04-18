from xmlrpc.client import Boolean
import pandas as pd
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u

import matplotlib.pyplot as plt
import seaborn as sns


from astropy.coordinates import match_coordinates_3d
from astropy import coordinates


def colnames():

    # Словарь для унификации имен колонок

    srg_names = {
                'id_src_name': 'srcname_fin',  # Индексы рентгеновских источников
                'x_ra_name': 'RA_fin',  # Координаты рентгеновских источников
                'x_dec_name': 'DEC_fin',
                'dl_name': 'DET_LIKE_0',  # Detection Likelihood
                'x_flux_name': 'flux_05-20',
                'ext_name': 'EXT_LIKE',  # Протяженность рентгеновских источников
                'ls_ra_name': 'ra',  # Координаты источников DESI
                'ls_dec_name': 'dec',
                'r_98_name': 'pos_r98',  # Позиционная ошибка
                'sigma_2d_name': 'pos_sigma_2d'
                }
    return srg_names


def pos_r_correction(df: pd.DataFrame, pos_r_cn: str ='pos_r98'):
    """
    TODO: redefine this function

    Corrects r98 (emperical calibration)

    Args:
        df (pd.DataFrame): Catalogue with r98 column.
        pos_r_cn (str, optional): r98 column name . Defaults to 'pos_r98'.
    """

    corr_coeff = 1.1
    lower_lim = 5
    new_pos_name = f'{pos_r_cn}_corr'
    
    df[new_pos_name] = df[pos_r_cn] * corr_coeff
    df[new_pos_name] = np.where(df[new_pos_name] > lower_lim,
                                df[new_pos_name],
                                lower_lim)

    print(f'{pos_r_cn} correction ({new_pos_name})')
    print('- ' * 5)
    print(f'Coefficient: {corr_coeff}')
    print(f'Lower threshold: {lower_lim}')
    print()

    return df


def xray_filtration(df: pd.DataFrame,
                    DL_thresh: float = 6,
                    EL_thresh: float = 6,
                    verbouse: Boolean=True) -> pd.DataFrame:
    """
    Filters X-ray sources.
    TODO: remake processing of duplicates
    """
    
    if verbouse:
        print(f'DET_LIKE_0 > {DL_thresh}')
        print(f'EXT_LIKE < {EL_thresh}')
        print()

        print(f'Before X-ray source filters: {len(df)}')

    df = df[(df['DET_LIKE_0'] > DL_thresh)&
            (df['EXT_LIKE'] < EL_thresh)]

    if verbouse:
        print(f'After X-ray source filters: {len(df)}')
        print()

    df = pos_r_correction(df)

    # Manually get rid of faint sources in duplicated pairs
    df = df[~((df['srcname_fin']=='SRGe J104659.3+573056')&(df['DET_LIKE_0'] < 20))]
    df = df[~((df['srcname_fin'] == 'SRGe J104700.7+574558')&(df['DET_LIKE_0'] < 20))]
    print('Weak ERO duplicates removed (temporary measure)')
    print()

    return df


def erosita_x_ray_filter(ero_df: pd.DataFrame,
                    DL_thresh: float = 6,
                    EL_thresh: float = 6) -> pd.DataFrame:
    """
    erosita_x_ray_filter cleans erosita catalog from sources with small detection likelihood and extended sources, also removes duplicates from the

    Args:
        ero_df (pd.DataFrame): DataFrame of eROSITA data from Lockman Hole. Normally it would be `ERO_lhpv_03_23_sd01_a15_g14_orig.pkl` file
        DL_thresh (float, optional): minimum Detection likelihood . Defaults to 6.
        EL_thresh (float, optional): Maximum extension likelihood. Defaults to 6.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned eROSITA data
    """  


    print(f'Number of sources after DL and EL cuts + duplicates removal: {len(ero_df)}')


    def cross_match_with_itself(xcat):
        xcat_matched = xcat.copy()
        c = SkyCoord(ra=xcat['RA_fin']*u.degree, dec=xcat['DEC_fin']*u.degree)
        catalog = SkyCoord(ra=xcat['RA_fin']*u.degree, dec=xcat['DEC_fin']*u.degree)
        idx, ero2ero, _ = c.match_to_catalog_sky(catalog, nthneighbor=2)
        ero2ero = ero2ero.to(u.arcsec).value

        xcat_matched['sep_to_closest'] = ero2ero
        xcat_matched.loc[:, 'srcname_fin_closest']  = xcat_matched.iloc[idx]['srcname_fin'].values
        xcat_matched = xcat_matched.merge(xcat.rename(columns={'srcname_fin':'srcname_fin_closest'}), on='srcname_fin_closest', how='left', suffixes=('', '_closest'))

        xcat_matched['is_confused'] = xcat_matched.eval('sep_to_closest<10') # close pairs for sep < 10. They are marked and the brightest one is kept

        xcat_matched['ML_FLUX_0_ratio'] = xcat_matched.ML_FLUX_0/xcat_matched.ML_FLUX_0_closest
        xcat_matched['ML_CTS_0_ratio']  = xcat_matched.ML_CTS_0/xcat_matched.ML_CTS_0_closest
        xcat_matched['DET_LIKE_0_ratio'] = xcat_matched.DET_LIKE_0/xcat_matched.DET_LIKE_0_closest
        xcat_matched['sep_ero2ero'] = xcat_matched['sep_to_closest']
        xcat_matched['pos_r98_first'] = xcat_matched['pos_r98']
        xcat_matched['pos_r98_second'] = xcat_matched['pos_r98_closest']
        xcat_matched['should_be_deleted'] = (xcat_matched['is_confused']) &  (xcat_matched['ML_FLUX_0_ratio']<1) #so that we delete the one with lower ML_FLUX_0


        xcat_matched = xcat_matched[['srcname_fin', 'srcname_fin_closest', 'is_confused', 'ML_FLUX_0_ratio', 'ML_CTS_0_ratio', 'DET_LIKE_0_ratio', 'sep_ero2ero', 'pos_r98_first', 'pos_r98_second', 'should_be_deleted']]

        return xcat_matched


    ero_df = ero_df.copy()
    ero_df.reset_index(inplace=True, drop=True)
    print(f'Original number of sources: {len(ero_df)}')
    ero_df = ero_df.query(f'DET_LIKE_0>{DL_thresh} and EXT_LIKE<{EL_thresh}')
    print(f'Number of sources after DL and EL cuts: {len(ero_df)}')

    ero_df.sort_values(by='ML_FLUX_0', ascending=False, inplace=True)
    ero_df.drop_duplicates(subset=['srcname_fin'], inplace=True) #drop duplicated srcname from the catalog, keeping the one with the highest ML_FLUX_0


    #cross match with itself to remove duplicates
    ero_df_cross_matched = cross_match_with_itself(ero_df)
    id_to_retain = ero_df_cross_matched[ero_df_cross_matched.should_be_deleted==False]['srcname_fin']
    ero_df = ero_df[ero_df.srcname_fin.isin(id_to_retain)]

    print(f'Number of sources after DL and EL cuts + duplicates removal: {len(ero_df)}')
    ero_df.reset_index(drop=True, inplace=True) #drop index

    return ero_df



# xcat_orig = pd.read_pickle(data_path+'ERO_lhpv_03_23_sd01_a15_g14_orig.pkl')
# xcat_orig = xcat_orig.query('EXT==0')
# xcat_orig.sort_values(by='ML_FLUX_0', ascending=False, inplace=True)
# xcat_orig = xcat_orig.drop_duplicates(subset=['srcname_fin'])


# def cross_match_with_itself(xcat, ra_col='RA_fin', dec_col='DEC_fin', err_col='pos_r98'):
    
#     xcat_matched = xcat.copy()
#     c = SkyCoord(ra=xcat[ra_col]*u.degree, dec=xcat[dec_col]*u.degree)
#     catalog = SkyCoord(ra=xcat[ra_col]*u.degree, dec=xcat[dec_col]*u.degree)
#     idx, ero2ero, _ = c.match_to_catalog_sky(catalog, nthneighbor=2)
#     ero2ero = ero2ero.to(u.arcsec).value

#     xcat_matched['sep_to_closest'] = ero2ero
#     xcat_matched.loc[:, 'srcname_fin_closest'] = xcat_matched.iloc[idx]['srcname_fin'].values
#     xcat_matched = xcat_matched.merge(xcat.rename(columns={
#                                       'srcname_fin': 'srcname_fin_closest'}),
#                                       on='srcname_fin_closest', how='left', suffixes=('', '_closest'))

#     xcat_matched['is_confused'] = xcat_matched.eval(
#         'sep_to_closest<sqrt( pos_r98**2 + pos_r98_closest**2 )/2 & sep_to_closest<30')

#     xcat_matched['ML_FLUX_0_ratio'] = xcat_matched.ML_FLUX_0 / \
#         xcat_matched.ML_FLUX_0_closest

#     xcat_matched['ML_CTS_0_ratio'] = xcat_matched.ML_CTS_0 / \
#         xcat_matched.ML_CTS_0_closest

#     xcat_matched['DET_LIKE_0_ratio'] = xcat_matched.DET_LIKE_0 / \
#         xcat_matched.DET_LIKE_0_closest

#     xcat_matched['sep_ero2ero'] = xcat_matched['sep_to_closest']
#     xcat_matched['pos_r98_first'] = xcat_matched['pos_r98']
#     xcat_matched['pos_r98_second'] = xcat_matched['pos_r98_closest']
#     xcat_matched['should_be_deleted'] = (xcat_matched['is_confused']) &(
#                                          xcat_matched['ML_FLUX_0_ratio'] < 1)
#                                         # so that we delete the one with lower ML_FLUX_0

#     xcat_matched = xcat_matched[['srcname_fin', 'srcname_fin_closest', 'is_confused', 'ML_FLUX_0_ratio',
#                                  'ML_CTS_0_ratio', 'DET_LIKE_0_ratio', 'sep_ero2ero', 'pos_r98_first',
#                                  'pos_r98_second', 'should_be_deleted']]

#     return xcat_matched


# xcat_matched = cross_match_with_itself(xcat_orig)


# id_to_retain = xcat_matched[xcat_matched.should_be_deleted == False]['srcname_fin']
# xcat_orig = xcat_orig[xcat_orig.srcname_fin.isin(id_to_retain)]



def desi_preprocessing(desi_fits_path: str) -> pd.DataFrame:
    """
    Preprocess DESI catalogue after TOPCAT correlation.

    Args:
        desi_fits_path (str): Path to the DESI catalogue (.fits).

    Returns:
        pd.DataFrame: Preprocessed DESI catalogue.
    """

    lh_desi_init_df = Table.read(desi_fits_path)

    # Usefull columns
    desi_columns = ['srcname_fin', 'RA_fin', 'DEC_fin', 'sep', 'release', 'brickid',
                    'objid', 'brick_primary', 'type', 'ra', 'dec', 'ra_ivar', 'dec_ivar',
                    'flux_g', 'flux_r', 'flux_z', 'flux_w1', 'flux_w2', 'flux_w3', 'flux_w4',
                    'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z', 'flux_ivar_w1', 'flux_ivar_w2',
                    'flux_ivar_w3', 'flux_ivar_w4', 'pmra', 'pmdec', 'parallax',
                    'pmra_ivar', 'pmdec_ivar', 'parallax_ivar']

    lh_desi_init_df = lh_desi_init_df[desi_columns].to_pandas()
    lh_desi_init_df['srcname_fin'] = lh_desi_init_df['srcname_fin'].astype(str).str[2:-1]

    lh_desi_init_df['desi_id'] = lh_desi_init_df['release'].astype(str) + '_' + \
                                 lh_desi_init_df['brickid'].astype(str) + '_' + \
                                 lh_desi_init_df['objid'].astype(str)

    lh_desi_init_df = lh_desi_init_df[lh_desi_init_df['brick_primary']==True]

    n_duplicates = lh_desi_init_df.duplicated(subset=['RA_fin', 'DEC_fin', 'ra', 'dec']).sum()
    print(f'Duplicates (by coordinates): {n_duplicates}')
    print()

    return lh_desi_init_df



def star_extraction(df: pd.DataFrame,
                    s_n_threshold:float,
                    optic_cat: str) -> pd.DataFrame:
    """
    Extracts stars from the DESI catalogue.

    Args:
        df (pd.DataFrame): DESI catalogue.
        s_n_threshold (float): Signal to noise threshold ratio.

    Returns:
        pd.DataFrame: DESI catalogue (only stars).
    """
    print('Extracting stars from Gaia catalogue...')
    print()

    print(f'S/N threshold: {s_n_threshold} (for parallax & propper motion)')
    print()

    init_shape = len(df)
    print(f'Before stars extraction: {init_shape}')

    if optic_cat=='gaia':
        df['parallax_sn'] = df['parallax'].abs() / df['parallax_error']
        df['pmra_sn'] = df['pmra'].abs() / df['pmra_error']
        df['pmdec_sn'] = df['pmdec'].abs() / df['pmdec_error']

    elif optic_cat=='desi':
        df['parallax_sn'] = df['parallax'] * np.sqrt(df['parallax_ivar'])
        df['pmra_sn'] = df['pmra'] * np.sqrt(df['pmra_ivar'])
        df['pmdec_sn'] = df['pmdec'] * np.sqrt(df['pmdec_ivar'])

    else:
        raise ValueError('`optic_cat` must be either "gaia" or "desi"')

    df = df[((df['parallax_sn'] > s_n_threshold) |
             (df['pmra_sn'] > s_n_threshold) |
             (df['pmdec_sn'] > s_n_threshold))]

    print(f'After stars extraction: {len(df)} ({len(df) / init_shape:.0%})')
    print()

    return df


def pos_r_filter(df: pd.DataFrame, sep_cname: str) -> pd.DataFrame:
    """
    Max separation constrain.

    Args:
        df (pd.DataFrame): X-ray catalogue.
        sep_cname (str): Column name for the separation.

    Returns:
        pd.DataFrame: X-ray catalogue with max separation constraint.
    """

    print(f'Before max separation filter: {len(df)}')
    
    df = df[df[sep_cname] < df['pos_r98']]

    print(f'After max separation filter: {len(df)}')
    print()

    return df


def qso_marker(qso_df: pd.DataFrame,
               other_cat_df: pd.DataFrame,
               optic_ra_name: str = 'ra',
               optic_dec_name: str = 'dec'):
    """
    Correlate QSO catalog with other catalog.

    Args:
        qso_df (pd.DataFrame): QSO catalogue
        other_cat_df (pd.DataFrame): Other catalogue
    """

    # Define coordinates
    ra_other_cpart, dec_other_cpart = other_cat_df[optic_ra_name], other_cat_df[optic_dec_name]
    ra_qso, dec_qso = qso_df['RA'], qso_df['DEC']

    cpart_coord = SkyCoord(ra=ra_other_cpart*u.degree, dec=dec_other_cpart*u.degree)
    drq_coord = SkyCoord(ra=ra_qso*u.degree, dec=dec_qso*u.degree)

    # Correlate
    RADIUS = 1.0
    print(f'Correlating other candidates with SDSS QSOs in {RADIUS}"...')
    print()
    MAX_SEP = RADIUS * u.arcsec
    idx, d2d, _ = cpart_coord.match_to_catalog_3d(drq_coord)
    sep_constraint = d2d < MAX_SEP
    cpart_matches = cpart_coord[sep_constraint]
    drq_catalog_matches = drq_coord[idx[sep_constraint]]

    # W1 magnitude extraction
    radec_data = {'RA': drq_catalog_matches.ra.deg, 'DEC': drq_catalog_matches.dec.deg}
    drq_radec_df = pd.DataFrame(data=radec_data)

    drq_w1_df = drq_radec_df.merge(qso_df[['RA', 'DEC', 'W1_MAG', 'W1_Jy_qso', 'Z']],
                                   on=['RA', 'DEC'], how='left')

    # QSO coordinates
    radec_data = {optic_ra_name: cpart_matches.ra.deg,
                  optic_dec_name: cpart_matches.dec.deg}
    cpart_qso_radec_df = pd.DataFrame(data=radec_data)

    cpart_qso_radec_df['is_qso'] = True
    cpart_qso_radec_df['W1_MAG'] = drq_w1_df['W1_MAG']
    cpart_qso_radec_df['W1_Jy_qso'] = drq_w1_df['W1_Jy_qso']
    cpart_qso_radec_df['Z'] = drq_w1_df['Z']

    cpart_qso_radec_df = cpart_qso_radec_df.drop_duplicates()

    print('*   ' * 5)
    stat_str = f'Отмечено {cpart_qso_radec_df.shape[0]} квазаров'
    print(stat_str + f' среди всех кандидатов DESI в r_98 ({other_cat_df.shape[0]})')
    print('*   ' * 5)

    # Mark quasars in r_false cpart sample
    desi_csc_drq_df = other_cat_df.merge(cpart_qso_radec_df,
                                     on=[optic_ra_name, optic_dec_name],
                                     how='left')

    desi_csc_drq_df['is_qso'] = desi_csc_drq_df['is_qso'].fillna(False)
    # cpart_qso_df = desi_csc_drq_df[desi_csc_drq_df['is_qso']==True]

    return desi_csc_drq_df


def gaia_coord_correction(
        df: pd.DataFrame,
        s_n_threshold: int,
        epochs_dict: dict = {'desi_epoch': 2017, 'ero_epoch': 2019.8},
        recalculate_separation = True
        ) -> pd.DataFrame:
    """
    Corrects Gaia coordinates for the proper motion.

    ra  + 1. / 3600e3 * pmra_gaia  * (epoch - 2015.5) / cos(radians(dec)) as ra_gaia,
    dec + 1. / 3600e3 * pmdec_gaia * (epoch - 2015.5) as dec_gaia

    ref: https://www.cosmos.esa.int/web/gaia-users/archive/combine-with-other-data

    Args:
        df (pd.DataFrame): Gaia catalogue.
        epoch (float): Epoch for which coordinates will be propagated.
        s_n_threshold (int): S/N threshold for pmra and pmdec.
        epochs_dict (dict): Dictionary with epochs for different surveys.

    Returns:
        pd.DataFrame: Gaia catalogue with corrected coordinates for all specified epochs.
    """
    gaia_epoch = 2016

    print(f'Gaia epoch: {gaia_epoch}')

    # Select only reliable pmra and pmdec (allows avoiding correction on noise)
    pmra_sn = df['pmra'].abs() / df['pmra_error']
    df['reliable_pmra'] = np.where(pmra_sn > s_n_threshold, df['pmra'], np.nan)

    pmdec_sn = df['pmdec'].abs() / df['pmdec_error']
    df['reliable_pmdec'] = np.where(pmdec_sn > s_n_threshold, df['pmdec'], np.nan)

    for postfix, epoch in epochs_dict.items():

        print()
        print(f'Target epoch: {epoch}')

        ra_prop_name = f'ra_{postfix}'
        dec_prop_name = f'dec_{postfix}'

        # Positional corrections for objrcts with reliable proper motion measurements
        df[ra_prop_name] = df['ra']  + df['reliable_pmra'] / 3600e3  * (epoch - gaia_epoch)\
                                    / np.cos(np.radians(df['dec']))
        df[dec_prop_name] = df['dec'] + df['reliable_pmdec'] / 3600e3 * (epoch - gaia_epoch)

        # Fill all NaN with original coordinates
        df[ra_prop_name] = df[ra_prop_name].fillna(df['ra'])
        df[dec_prop_name] = df[dec_prop_name].fillna(df['dec'])

        if recalculate_separation:
            # Recalculate separation
            ero_coord = SkyCoord(df['RA_fin']*u.deg, df['DEC_fin']*u.deg, frame='icrs')
            gaia_coord_prop = SkyCoord(df[ra_prop_name]*u.deg,
                                    df[dec_prop_name]*u.deg, frame='icrs')
            sep_prop = ero_coord.separation(gaia_coord_prop)
            df[f'sep_{postfix}'] = sep_prop.arcsec
            print(f'sep_{postfix}')

        print(f'Column names: {ra_prop_name}, {dec_prop_name}')

    return df


"""
TODO: Перенести "преобразующие" функции (см. ниже) в отдельный модуль.
"""

def flux2mag(flux):

    return 22.5 - 2.5 * np.log10(flux)


def flux_nmagg2vega_mag(flux:pd.Series,
                        mode:str) -> pd.Series:
    """
    Converts DESI w1 flux (in nanomaggies) to
    vega magnitudes.
    
    https://www.legacysurvey.org/dr9/description/
    """
    if mode=='w1':
        delta_m = 2.699
    elif mode=='w2':
        delta_m = 3.339
    elif mode=='w3':
        delta_m = 5.174
    elif mode=='w4':
        delta_m = 6.620
    
    vega_flux = flux * 10 ** (delta_m / 2.5)
    vega_mag = flux2mag(vega_flux)
    vega_mag = vega_mag.replace([np.inf, -np.inf], np.nan)
    
    return vega_mag


def flux_frequency_correction(magnitudes: pd.Series,
                              w_eff: float,
                              zeropoint: float) -> pd.Series:
    """
    Converts magnitudes obtainded from nanomaggies (erg/cm²/Hz)
    to fluxes in erg/cm²/s.

    http://svo2.cab.inta-csic.es/theory/fps/index.php?id=KPNO/MzLS.z&&mode=browse&gname=KPNO&gname2=MzLS#filter
    http://svo2.cab.inta-csic.es/theory/fps/index.php?id=GAIA/GAIA3.G&&mode=browse&gname=GAIA&gname2=GAIA3#filter

    Args:
        magnitudes (pd.Series): Magnitudes in AB/Vega system.
        w_eff (float): width of the effective wavelength.
        zeropoint (float): Zero Point in AB/Vega System.

    Returns:
        pd.Series: flux in erg/cm²/s.
    """

    flux = w_eff * zeropoint * 10 ** (-0.4 * magnitudes)
    flux.name = 'flux_corrected'

    return flux


def reliable_magnitudes(df: pd.DataFrame,
                        s_n_threshold: int,
                        xray: bool=True,
                        colors: bool=True,
                        sb: bool=False) -> pd.DataFrame:
    """
    Calculate reliable magnitudes only for objects with reliable flux measurments.
    
    https://www.legacysurvey.org/dr9/description/:
    "The fluxes can be negative for faint objects, and indeed we expect
    many such cases for the faintest objects."

    Args:
        df (pd.DataFrame): DESI catalogue.
        s_n_threshold (int): S/N threshold. 4 by default.
        xray (bool): If True, calculate X-ray to optical flux ratio.
        colors (bool): If True, calculate colors.
        sb (bool): If True, adds prefix 'desi_' to desi columns.

    Returns:
        pd.DataFrame: Catalogue with reliable magnitudes.
    """
    
    for band in ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']:

        flux_colname = f'flux_{band}'
        flux_ivar_colname = f'flux_ivar_{band}'
        dered_mag_colname = f'dered_mag_{band}'

        if sb:
            prefix = 'desi_'
        else:
            prefix = ''

        # All magnitudes (unreliable included)
        df[f'all_mag_{band}'] = flux2mag(df[prefix + flux_colname])
        
        # Select only reliable fluxes (allows avoiding correction on noise)
        flux_sn = df[prefix + flux_colname].abs() * np.sqrt(df[prefix + flux_ivar_colname])
        reliable_flux = pd.Series(np.where(flux_sn > s_n_threshold, df[prefix + flux_colname], np.nan))
        # df[f'rel_flux_{band}'] = reliable_flux

        # Calculate reliable magnitudes
        df[f'rel_mag_{band}'] = flux2mag(reliable_flux)
        df[f'rel_mag_{band}'] = df[f'rel_mag_{band}'].replace([np.inf, -np.inf], np.nan)

        # Reliable dereddended magnitudes
        df[f'rel_dered_mag_{band}'] = np.where(
            flux_sn > s_n_threshold, df[prefix + dered_mag_colname], np.nan
            )

        # Reliable Vega magnitudes for WISE fluxes
        if 'w' in band:
            df[f'vega_mag_{band}'] = flux_nmagg2vega_mag(reliable_flux, mode=band)

    W_EFF_G = 1204.22
    AB_ZEROPOINT_G = 4.78525e-9

    df['rel_desi_flux_corr_g'] = flux_frequency_correction(
            df['desi_rel_dered_mag_g'],
            w_eff=W_EFF_G,
            zeropoint=AB_ZEROPOINT_G
            )

    W_EFF_R = 1311.48
    AB_ZEROPOINT_R = 2.66574e-9

    df['rel_desi_flux_corr_r'] = flux_frequency_correction(
            df['desi_rel_dered_mag_r'],
            w_eff=W_EFF_R,
            zeropoint=AB_ZEROPOINT_R
            )

    W_EFF_Z = 1291.48
    AB_ZEROPOINT_Z = 1.286e-9

    df['rel_desi_flux_corr_z'] = flux_frequency_correction(
            df['desi_rel_dered_mag_z'],
            w_eff=W_EFF_Z,
            zeropoint=AB_ZEROPOINT_Z
            )

    if colors:
        # Colors
        df['rel_g_r'] = df['rel_mag_g'] - df['rel_mag_r']
        df['rel_g_z'] = df['rel_mag_g'] - df['rel_mag_z']
        df['rel_r_z'] = df['rel_mag_r'] - df['rel_mag_z']

        df['all_g_r'] = df['all_mag_g'] - df['all_mag_r']
        df['all_g_z'] = df['all_mag_g'] - df['all_mag_z']
        df['all_r_z'] = df['all_mag_r'] - df['all_mag_z']

        df['dered_g_r'] = df[prefix + 'dered_mag_g'] - df[prefix + 'dered_mag_r']
        df['dered_g_z'] = df[prefix + 'dered_mag_g'] - df[prefix + 'dered_mag_z']
        df['dered_r_z'] = df[prefix + 'dered_mag_r'] - df[prefix + 'dered_mag_z']

        df['rel_dered_g_r'] = df['rel_dered_mag_g'] - df['rel_dered_mag_r']
        df['rel_dered_g_z'] = df['rel_dered_mag_g'] - df['rel_dered_mag_z']
        df['rel_dered_r_z'] = df['rel_dered_mag_r'] - df['rel_dered_mag_z']

        df['rel_g_w1'] = df['rel_mag_g'] - df['rel_mag_w1']
        df['rel_r_w1'] = df['rel_mag_r'] - df['rel_mag_w1']
        df['rel_z_w1'] = df['rel_mag_z'] - df['rel_mag_w1']
        
        df['rel_g_w2'] = df['rel_mag_g'] - df['rel_mag_w2']
        df['rel_r_w2'] = df['rel_mag_r'] - df['rel_mag_w2']
        df['rel_z_w2'] = df['rel_mag_z'] - df['rel_mag_w2']

        df['rel_w1_w2'] = df['rel_mag_w1'] - df['rel_mag_w2']
        df['rel_w2_w3'] = df['rel_mag_w2'] - df['rel_mag_w3']

        df['vega_w1_w2'] = df['vega_mag_w1'] - df['vega_mag_w2']
        df['vega_w2_w3'] = df['vega_mag_w2'] - df['vega_mag_w3']

        df['rel_w1_w3'] = df['rel_mag_w1'] - df['rel_mag_w3']
        df['rel_w1_w4'] = df['rel_mag_w1'] - df['rel_mag_w4']

        df['vega_w1_w3'] = df['vega_mag_w1'] - df['vega_mag_w3']
        df['vega_w1_w4'] = df['vega_mag_w1'] - df['vega_mag_w4']

    if xray:
        # X-ray to optical flux
        df['lg(Fx/Fo_g)'] = np.log10(df['flux_05-20'] / df[prefix + 'flux_g'])
        df['lg(Fx/Fo_r)'] = np.log10(df['flux_05-20'] / df[prefix + 'flux_r'])
        df['lg(Fx/Fo_z)'] = np.log10(df['flux_05-20'] / df[prefix + 'flux_z'])

        '''
        TODO: update with datalab data when possible
        '''

        dered_flux_z = 10 ** (9 - df['rel_dered_mag_z'] / 2.5)
        df['rel_dered_lg(Fx/Fo_z)'] = np.log10(df['flux_05-20'] / dered_flux_z)
        df['rel_dered_lg(Fx/Fo_z_corr)'] = np.log10(df['flux_05-20'] / df['rel_desi_flux_corr_z'])

        dered_flux_g = 10 ** (9 - df['rel_dered_mag_g'] / 2.5)
        df['rel_dered_lg(Fx/Fo_g)'] = np.log10(df['flux_05-20'] / dered_flux_g)

        dered_flux_r = 10 ** (9 - df['rel_dered_mag_r'] / 2.5)
        df['rel_dered_lg(Fx/Fo_r)'] = np.log10(df['flux_05-20'] / dered_flux_r)

    return df


def ab_mag2Jy(mag):
    '''
    Converts AB magnitude to Jy.
    '''
    return 10 ** (3.56 - 0.4 * mag)


def leave_closest_sep(df: pd.DataFrame, sep_name: str,
                      src_id_name: str = 'srcname_fin') -> pd.DataFrame:
    """
    Leaves only closest objects in the catalogue.

    Args:
        df (pd.DataFrame): Catalogue.
        sep_name (str): Separation name.

    Returns:
        pd.DataFrame: Catalogue with only closest objects.
    """

    closest_idx = df.groupby(src_id_name)[sep_name].idxmin()
    closest_df = df.loc[closest_idx]

    return closest_df


# def poisson_k(k: int, density: float, r98_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Функция для вычисления вероятности обнаружить `k` объектов в
#     кружках с радиусом r98 при средней плотности `density` согласно
#     расспределению Пуассона.

#     Args:
#         k (int): Количество объектов в кружках с радиусом r98 (arcsec).
#         density (float): Средняя плотность (в arcsec^-2).
#         r98_df (pd.DataFrame): Таблица с кандидатами в пределах r98.
#     """    

#     print(f'** k = {k} **')

#     observed_counts_r98 = r98_df.groupby('srcname_fin').size()

#     n_Gaia_idxs = observed_counts_r98[observed_counts_r98 == k].index.sort_values()

#     radii = (r98_df[r98_df['srcname_fin'].isin(n_Gaia_idxs)]
#             .sort_values('srcname_fin')['pos_r98_corr'].unique())

#     rows = []
#     for srcid, r in zip(n_Gaia_idxs, radii):

#         expected_n = np.pi * r ** 2 * density
#         probability = poisson.pmf(k, mu=expected_n) * 100
#         row = [srcid, r, probability]
#         rows.append(row)

#     df = pd.DataFrame(rows, columns=['srcname_fin', 'pos_r98_corr', 'probability (%)'])
#     df['pos_r98_corr'] = np.round(df['pos_r98_corr']).astype(int)
#     df = df.sort_values('pos_r98_corr', ascending=False).reset_index(drop=True)

#     return df


# def cat2hpx(ra, dec, nside):
#     """
#     Convert a catalogue to a HEALPix map of number counts per resolution
#     element.

#     Parameters
#     ----------
#     lon, lat : (ndarray, ndarray)
#         Coordinates of the sources in degree. If radec=True, assume input is in the icrs
#         coordinate system. Otherwise assume input is glon, glat

#     nside : int
#         HEALPix nside of the target map

#     radec : bool
#         Switch between R.A./Dec and glon/glat as input coordinate system.

#     Return
#     ------
#     hpx_map : ndarray
#         HEALPix map of the catalogue number counts in Galactic coordinates

#     """

#     npix = hp.nside2npix(nside)

#     # convert to HEALPix indices
#     indices = hp.ang2pix(nside, ra, dec, lonlat=True)

#     idx, counts = np.unique(indices, return_counts=True)

#     # fill the fullsky map
#     hpx_map = np.zeros(npix, dtype=int)
#     hpx_map[idx] = counts

#     return hpx_map, counts



'''
Classification
'''


# def star_slope(df: pd.DataFrame,
#                feature_names: list,
#                gaia_marker_cn: str) -> pd.DataFrame:
#     """
#     Classifies stars and not stars based on features.

#     Args:
#         df (pd.DataFrame): DataFrame with features.
#         feature_names (list): List of features names.
#         gaia_marker_cn (str): Gaia marker column name.
#     """

#     # Replase inf with nan (10 - 20 objects)
#     for col in feature_names:
#         df[col] = df[col].replace([np.inf, -np.inf], np.nan)

#     # Remove nan (10 - 30 objects)
#     df = df.dropna(subset=feature_names)

#     X = df[feature_names]
#     Y = df[gaia_marker_cn]

#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#     # Fit (train) the Logistic Regression classifier
#     lr_clf = linear_model.LogisticRegressionCV(cv=5,
#                                                class_weight='balanced',
#                                                scoring='roc_auc')
#     lr_clf.fit(X_train, y_train)

#     ra_metric = roc_auc_score(y_test, lr_clf.predict_proba(X_test)[:, 1])

#     y_pred_total = lr_clf.predict(X)
#     total_confusion = confusion_matrix(Y, y_pred_total, normalize='true')
#     total_confusion = np.round(total_confusion, 2)
#     # print(f'Total confusion matrix: \n{total_confusion}')

#     percent_matrix = np.round(total_confusion * 100).astype(int).astype(str)
#     percent_matrix_formatted = np.zeros((2, 2)).astype(str)
#     for (x,y), value in np.ndenumerate(percent_matrix):
#         percent_matrix_formatted[x, y] = f' ({value}%)'

#     total_confusion_number = confusion_matrix(Y, y_pred_total)
#     # print(f'Total confusion numbers: \n{total_confusion_number}')

#     total_confusion_number = np.round(total_confusion_number).astype(int).astype(str)
#     matrix_stat = np.char.add(total_confusion_number, percent_matrix_formatted)
#     # print(f'Total confusion matrix with percentages: \n{matrix_stat}')

#     w = lr_clf.coef_[0]
#     k = -w[0] / w[1]
#     b = - (lr_clf.intercept_[0]) / w[1]

#     def linear_func(x):
#         return k * x + b

#     return linear_func, ra_metric, matrix_stat


def plot_scatter(star_df: pd.DataFrame,
                 notstar_df: pd.DataFrame,
                 ys: list,
                 x: str,
                 classifiers: list,
                 matrices: list,
                 test_metrics: list,
                 mode: str = 'optical'):
    """
    Plot the scatter plot for stars and not stars
    with division line and confusion matrix.

    Args:
        star_df (pd.DataFrame): DataFrame with stars.
        notstar_df (pd.DataFrame): DataFrame with not stars
        ys (list): Vertical axe.
        x (str): Horizontal axe.
        classifiers (list): Pretrained classifiers.
        matrices (list): Confusion matrices.
        test_metrics (list): Test metrics.
    """

    xlims = np.array([-19, -13])
    ylims = [-3, 7]
    text_y = 6.2
    text_x = -18

    if mode=='flux_corr':
        xlims = np.array([-4, 2])
        text_x = -3

    i = 0

    _, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for ax, y, clf, matrix, r_a, in zip(axs, ys,
                                        classifiers,
                                        matrices,
                                        test_metrics):

        if y=='rel_w1_w2' or y=='vega_w1_w2':
            hist_lims = [-1.2, 1.5]
            w_space = np.linspace(*hist_lims, 60)
            _, ax0 = plt.subplots(figsize=(10, 2))
            ax0.hist(star_df[y], bins=w_space, histtype='step',
                     color='red', lw=3, label='star')
            ax0.hist(notstar_df[y], bins=w_space, histtype='step',
                     color='gray', lw=3, label='not star')
            ax0.set_xlim(*hist_lims)
            ax0.set_xlabel(y)
            ax0.legend(loc='upper left', fontsize=12)

        sns.scatterplot(data=star_df, x=x, y=y, ax=ax, s=20, linewidth=.8,
                        alpha=.7, edgecolor='r', color='none', zorder=2, label='star')

        # if i==0:
        #     print('STARS')
        #     print(star_df.sort_values(by=y)[['ra', 'dec']].head(50).reset_index(drop=True))

        #     print('NOT STARS')
        #     print(notstar_df.sort_values(by=y)[['ra', 'dec']].head(30).reset_index(drop=True))
        # sns.scatterplot(data=star_df, x=x, y=y, ax=ax, s=30, linewidth=1, zorder=3,
        #                 alpha=1, hue='spectraltype_esphs')
        
        sns.scatterplot(data=notstar_df, x=x, y=y, ax=ax, s=15,
                        alpha=.5, color='gray', zorder=1, label='not star')
        ax.set_ylim(*ylims)
        ax.set_xlim(*xlims)
        ax.plot(xlims, clf(xlims), color='black', ls='--')

        ax.text(text_x, text_y, f'Predicted label', fontsize=10)

        table = ax.table(cellText=matrix,
                        colLabels=['False', 'True'],
                        rowLabels=['False', 'True'],
                        edges='horizontal',
                        bbox=[.06, .705, .25, .19],
                        zorder=5)
        # table.set_fontsize(20)
        # table.scale(3, 3)

        ax.text(text_x - 1, .5 * text_y, f'Test ROC AUC: {r_a:.1%}', fontsize=10, alpha=.5)

        if i == 0:
            lgnd = ax.legend(frameon=True, loc='upper right', fontsize=12)
            for handle in lgnd.legendHandles:
                handle._sizes = [70]
        else:
            ax.legend('')

        i += 1

        if x=='rel_dered_lg(Fx/Fo_z_corr)' and y=='rel_dered_r_z':
            
            _, ax1 = plt.subplots(figsize=(10, 8))

            sns.scatterplot(data=star_df, x=x, y=y, ax=ax1, s=20, linewidth=.8,
                        alpha=.7, edgecolor='r', color='none', zorder=2, label='star')
        
            sns.scatterplot(data=notstar_df, x=x, y=y, ax=ax1, s=20,
                            alpha=.5, color='gray', zorder=1, label='not star')
            ax1.set_ylim(ylims[0], 4)
            ax1.set_xlim(*xlims)
            # ax1.plot(xlims, clf(xlims), color='black', ls='--')

            # ax1.text(.2, -1.6, f'Predicted label', fontsize=14)

            # table = ax1.table(
            #     cellText=matrix,
            #     colLabels=['False', 'True'],
            #     rowLabels=['False', 'True'],
            #     edges='horizontal',
            #     bbox=[.6, .05, .35, .13],
            #     zorder=5
            #     )
            # table.set_fontsize(12)
            # table.scale(3, 3)

            # ax1.text(1.3 * text_x, .6 * text_y, f'Test ROC AUC: {r_a:.1%}',
            #          fontsize=10, alpha=.5)

            ax1.set_title('MB, pm corrected')

            lgnd = ax1.legend(frameon=True, loc='upper right', fontsize=12)
            for handle in lgnd.legendHandles:
                handle._sizes = [70]

    plt.show()


def match_desi2gaia(desi_df: pd.DataFrame,
                    gaia_df: pd.DataFrame,
                    search_radius: float = 1.0,
                    sb: bool = False):
    
    """
    Mark stars in DESI dataframe based on Gaia star sample location

    Args:
        desi_df (pd.DataFrame): DESI dataframe (closest in r 98).
        gaia_df (pd.DataFrame): Gaia dataframe (only stars).
        search_radius (float): Search radius in arcsec.
        sb (bool): If True, adds prefix 'desi_' to desi column names.
    Returns:
        bool_markers (pd.Series): Boolean markers for each row in desi_df.
                                 If True, DESI soruce has a match in GAIA catalogue
                                 in `search radius`.
        closest_seps (np.array): Closest separation in arcsec for each source in desi_df.
        matched_df (pd.DataFrame): Dataframe with columns from both catalogues.
    """

    if sb:
        prefix = 'desi_'
    else:
        prefix = ''

    gaia_ra_name, gaia_dec_name = 'ra', 'dec'

    # Define coordinates
    ra_desi_cpart, dec_desi_cpart = desi_df[prefix + 'ra'], desi_df[prefix + 'dec']
    ra_gaia_star, dec_gaia_star = gaia_df[gaia_ra_name], gaia_df[gaia_dec_name]

    cpart_coord = SkyCoord(ra=ra_desi_cpart*u.degree, dec=dec_desi_cpart*u.degree)
    star_coord = SkyCoord(ra=ra_gaia_star*u.degree, dec=dec_gaia_star*u.degree)

    # Correlate
    print(f'Correlating catalogs in {search_radius}"...')
    MAX_SEP = search_radius * u.arcsec
    idx, d2d, _ = cpart_coord.match_to_catalog_3d(star_coord)
    sep_constraint = d2d < MAX_SEP
    cpart_matches = cpart_coord[sep_constraint]
    star_matches = star_coord[idx[sep_constraint]]

    _, closest_seps, _ = match_coordinates_3d(star_coord, cpart_coord, nthneighbor=1)

    # Desi coordinates table
    desi_radec = {'ra_desi': cpart_matches.ra.deg, 'dec_desi': cpart_matches.dec.deg}
    desi_radec_df = pd.DataFrame(data=desi_radec)

    # According Gaia coordinates table
    gaia_radec = {'ra_gaia': star_matches.ra.deg, 'dec_gaia': star_matches.dec.deg}
    gaia_radec_df = pd.DataFrame(data=gaia_radec)

    # Horizontal concatination of both (coordinate dataframe)
    matched_df = pd.concat([desi_radec_df, gaia_radec_df], axis=1)
    matched_df = matched_df.drop_duplicates()

    # Add desi columns to coordinate dataframe
    matched_df = matched_df.merge(
        desi_df[['desi_id', prefix + 'ra', prefix + 'dec']],
        right_on=[prefix + 'ra', prefix + 'dec'],
        left_on=['ra_desi', 'dec_desi'], how='left'
        )
    
    # Clean up
    matched_df = matched_df.drop(columns=[prefix + 'ra', prefix + 'dec'])
    # TODO: check if this is needed
    matched_df = matched_df.drop_duplicates()

    # Similarly for Gaia columns
    gaia_features = ['source_id',
                     gaia_ra_name, gaia_dec_name,
                     'spectraltype_esphs',
                     'pmra', 'pmdec', 'parallax',
                     'pmra_error', 'pmdec_error', 'parallax_error']
    matched_df = matched_df.merge(gaia_df[gaia_features],
                                  right_on=[gaia_ra_name, gaia_dec_name],
                                  left_on=['ra_gaia', 'dec_gaia'], how='left')
    matched_df = matched_df.drop(columns=['ra', 'dec'])
    # TODO: check if this is needed
    matched_df = matched_df.drop_duplicates()

    # Mark DESI object as a star if in has Gaia star in `serach radius`
    bool_markers = desi_df[prefix + 'ra'].isin(desi_radec_df['ra_desi'])

    return bool_markers, closest_seps.arcsec, matched_df


def metaplot(df: pd.DataFrame,
             x: str, ys: str,
             star_marker: str,
             band_mode: str = 'optical'
             ):
    """
    Plot the scatter plot for stars and not stars for three different colors.

    Args:
        df (pd.DataFrame): Preprocessed dataframe.
        x (str): Horizontal axe.
        ys (str): Vertical axes.
        star_marker (str): Star marker.
        band_mode (str, optional): Either optical or infrared plot parameters.
                                   Defaults to 'optical'.
    """

    desi_gaia_star_df = df[df[star_marker]==True]
    desi_gaia_notstar_df = df[df[star_marker]==False]

    # Deviding line function, test metric, confusion matrix
    clf_z2_gr, test_z2_gr, cnfsn_z2_gr = star_slope(df, [x, ys[0]],
                                                    gaia_marker_cn=star_marker)
    clf_z2_gz, test_z2_gz, cnfsn_z2_gz = star_slope(df, [x, ys[1]],
                                                    gaia_marker_cn=star_marker)
    clf_z2_rz, test_z2_rz, cnfsn_z2_rz = star_slope(df, [x, ys[2]],
                                                    gaia_marker_cn=star_marker)

    plot_scatter(
        star_df=desi_gaia_star_df,
        notstar_df=desi_gaia_notstar_df,
        ys=ys,
        x=x,
        classifiers=[clf_z2_gr, clf_z2_gz, clf_z2_rz],
        matrices=[cnfsn_z2_gr, cnfsn_z2_gz, cnfsn_z2_rz],
        test_metrics=[test_z2_gr, test_z2_gz, test_z2_rz],
        mode=band_mode)


def logNlogS(fluxes):
    
    flux_thresholds = np.logspace(-15, -12, 500)
    sln = np.array([len(fluxes[fluxes > x]) for x in flux_thresholds])

    return flux_thresholds, sln


def cross_match_data_frames(df1: pd.DataFrame, df2: pd.DataFrame, 
                            colname_ra1: str, colname_dec1: str,
                            colname_ra2: str, colname_dec2: str,
                            match_radius: float = 3.0,
                            df_prefix: str = '',
                            closest: bool = False,
                            solo_near: bool = False,
                            ero_sep: bool = False,
                            xray_ra='RA_fin',
                            xray_dec='DEC_fin'
                            ):
    """
    cross_match_data_frames cross-matches two dataframes.
    Cross-match two dataframes with astropy
    https://docs.astropy.org/en/stable/api/astropy.coordinates.match_coordinates_sky.html#astropy.coordinates.match_coordinates_sky
    https://docs.astropy.org/en/stable/api/astropy.coordinates.search_around_sky.html#astropy.coordinates.search_around_sky
    Args:
        df1 (pd.DataFrame): first catalog
        df2 (pd.DataFrame): second catalog
        colname_ra1 (str): columns name for ra in df1
        colname_dec1 (str): columns name for dec in df1
        colname_ra2 (str): columns name for ra in df2
        colname_dec2 (str): columns name for dec in df2
        match_radius (float, optional): match radius in arcsec. Defaults to 3.0.
        df_prefix (str, optional): prefix to prepend to the columns of the second data frame. Defaults to ''. If exists, '_' is appended.
        closest (bool, optional): whether to return the closest match. Defaults to False.
    Returns:
        pd.DataFrame: match of df1 and df2
        the columns are from the original df1 and df2 (with the prefix for df2). 
        added columns: 
        
        sep - separation in arcsec
        
        n_near - number of matches from df2 for a particular source from df1.
        For example n_near=10 for a source in df1 means that there are 10 sources
        in df2 within the match_radius.
        
        n_matches - number of entries of a source in a final table. For example,
        if some soruce lying between two X-ray soruces and falling into the
        match_radius for both of them, n_matches=2.
    
    example:
    cross_match_data_frames(
        desi,
        gaia, 
        colname_ra1='RA_fin',
        colname_dec1='DEC_fin',
        colname_ra2='ra',
        colname_dec2='dec',
        match_radius = 10,
        df_prefix = 'GAIA',
        closest=False
        )
    """
    if df_prefix != '':
        df_prefix = df_prefix + '_'
    else:
        df_prefix = ''

    df1 = df1.copy()
    df2 = df2.copy()

    orig_size_1 = df1.shape[0]
    orig_size_2 = df2.shape[0]

    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    df1.rename(columns={'index': 'index_primary'}, inplace=True)
    df2.rename(columns={'index': 'index_secondary'}, inplace=True)

    coords1 = SkyCoord(ra = df1[colname_ra1].values*u.degree, dec = df1[colname_dec1].values*u.degree)
    coords2 = SkyCoord(ra = df2[colname_ra2].values*u.degree, dec = df2[colname_dec2].values*u.degree)

    idx1, idx2, ang_sep, _ = coordinates.search_around_sky(coords1, coords2, match_radius*u.arcsec)
    ang_sep = ang_sep.to(u.arcsec)
    ang_sep = pd.DataFrame({df_prefix+'sep': ang_sep})

    df1 = df1.loc[idx1]
    df2 = df2.loc[idx2]

    df1.reset_index(inplace = True, drop = True)
    df2.reset_index(inplace = True, drop = True)

    df2.columns  = [df_prefix + x for x in df2.columns]
    df2.rename(columns={df_prefix+'index_secondary':'index_secondary'}, inplace=True)

    df_matched = pd.concat([df1, df2, ang_sep], axis=1)

    if ero_sep:

        # Separation between ERO sources and sources from the second catalog
        ero_cpart_coords = SkyCoord(
            ra=df_matched[xray_ra].values*u.degree, dec=df_matched[xray_dec].values*u.degree
            )
        prefix_cpart_coords = SkyCoord(
            ra = df_matched[df_prefix + colname_ra2].values*u.degree,
            dec = df_matched[df_prefix + colname_dec2].values*u.degree
            )
        df_matched[f'{df_prefix}ERO_sep'] = ero_cpart_coords.separation(prefix_cpart_coords).arcsec

    df_matched.sort_values(by=['index_primary', df_prefix+'sep'], inplace=True, ascending=True)

    df_matched[df_prefix + 'n_near'] = (
        df_matched.groupby('index_primary')[df_prefix+'sep'].transform('count')
        )

    if solo_near:
        df_matched = df_matched[df_matched[df_prefix+'n_near'] == 1]

    second_index_value_counts = df_matched['index_secondary'].value_counts()
    df_matched[df_prefix + 'n_matches'] = (
        df_matched['index_secondary'].apply(lambda x: second_index_value_counts[x])
        )

    print('cross-match radius:', match_radius, 'arcsec', '\n')
    print('total matches:', len(df_matched), 'out of', orig_size_1, 'x', orig_size_2)

    print('\t total unique pairs:', len(df_matched.query(df_prefix+'n_matches == 1')))
    print('\t total non-unique pairs (duplicates in df2):', len(df_matched.query(df_prefix+'n_matches > 1')), '\n')

    if closest:
        df_matched = df_matched.drop_duplicates(subset=['index_primary'], keep='first')
        print('total closest matches:', len(df_matched), '\n')

    df_matched.drop(columns=['index_primary'], inplace=True)
    df_matched.drop(columns=['index_secondary'], inplace=True)

    return df_matched


def closer_match(
    desi_smth_df,
    full_smth_df,
    df_prefix,
    colname_id2,
    colname_ra2,
    colname_dec2
    ):

    """
    Find sources that are closer to the ERO than DESI's spectral/pm counterpart.

    Args:
        desi_smth_df: Pre-correlated spectral/pm catalog.
        df_prefix: Prefix of the columns to be used.
        colname_ra2: RA column name of the full catalog.
        colname_dec2: DEC column name of the full catalog.
    """    
    
    # Slim version of the desi_smth catalog
    desi_smth_slim_df = desi_smth_df[
        ['srcname_fin', 'RA_fin', 'DEC_fin',
        f'{df_prefix}_{colname_ra2}',
        f'{df_prefix}_{colname_dec2}',
        f'{df_prefix}_{colname_id2}', f'{df_prefix}_ERO_sep']
        ]

    # Search for the closest match in the full smth catalog
    desi_smth_ero_closest_df = cross_match_data_frames(
        desi_smth_slim_df,
        full_smth_df[[colname_id2, colname_ra2, colname_dec2]],
        colname_ra1='RA_fin',
        colname_dec1='DEC_fin',
        colname_ra2=colname_ra2,
        colname_dec2=colname_dec2,
        match_radius = 15,
        df_prefix = f'closest_{df_prefix}',
        closest=True
        )

    # Compare the closest ERO match with the DESI spectral/pm match
    closer_match_df = desi_smth_ero_closest_df[
        desi_smth_ero_closest_df[f'closest_{df_prefix}_sep'] < desi_smth_ero_closest_df[f'{df_prefix}_ERO_sep']
        ]

    # Closest coordinates
    closer_coord = SkyCoord(
        ra=closer_match_df[f'closest_{df_prefix}_{colname_ra2}']*u.degree,
        dec=closer_match_df[f'closest_{df_prefix}_{colname_dec2}']*u.degree
        )

    # DESI coordinates
    desi_coord = SkyCoord(
        ra=closer_match_df[f'{df_prefix}_{colname_ra2}']*u.degree,
        dec=closer_match_df[f'{df_prefix}_{colname_dec2}']*u.degree,
    )
    
    closer_match_df[f'{df_prefix}_cpart_closest_sep'] = closer_coord.separation(desi_coord).arcsec

    print()
    print(f'{len(closer_match_df)} sources in {df_prefix} catalog are closer to ERO than DESI {df_prefix} counterpart.')

    return closer_match_df


# def rss(arr_1, arr_2):
#     """
#     Root sum square of two arrays.
#     """
#     return np.sqrt(arr_1**2 + arr_2**2)


def star_marker(df: pd.DataFrame, s_n_threshold: float, SB=True) -> pd.DataFrame:
    """
    Mark stars in a dataframe (both for positive and negative
    parallaxes/proper motions)

    Args:
        df (pd.DataFrame): dataframe with parallax and proper motion
        s_n_threshold (float): threshold for the signal-to-noise ratio

    Returns:
        pd.DataFrame: dataframe with stars marked
    """

    if SB:
        prefix = 'GAIA_'
        n_matches = np.sum(~df[f'{prefix}source_id'].isna())
        print((f'{n_matches} mathces with GAIA'), '\n')

    else:
        prefix = ''

    df[f'{prefix}parallax_sn'] = df[f'{prefix}parallax'] / df[f'{prefix}parallax_error']
    df[f'{prefix}pmra_sn'] = df[f'{prefix}pmra'].abs() / df[f'{prefix}pmra_error']
    df[f'{prefix}pmdec_sn'] = df[f'{prefix}pmdec'].abs() / df[f'{prefix}pmdec_error']
    df[f'{prefix}pm_sn'] = df[f'{prefix}pm'] / df[f'{prefix}pm_error']

    star_mask = (
        (df[f'{prefix}parallax_sn'] > 0) &
        (
            (df[f'{prefix}parallax_sn'] > s_n_threshold) |
            (df[f'{prefix}pmra_sn'] > s_n_threshold) |
            (df[f'{prefix}pmdec_sn'] > s_n_threshold) |
            (df[f'{prefix}pm_sn'] > s_n_threshold)
            )
        )

    print(np.sum(star_mask), 'stars found')

    result_colname = 'class_GAIA_class'

    df.loc[star_mask, result_colname] = 'GALACTIC'
    print(f'Markers are written in the {result_colname} column')

    return df


def calc_lumin(Fx, z):
    """
    Calculate luminosity from flux and redshift.

    Args:
        Fx (float): flux in 0.5-2 keV band
        z (float): redshift

    Returns:
        float: luminosity in erg/s
    """    

    lumin = 4 * np.pi * Fx * (u.erg/u.s/u.cm**2) * (cosmo.luminosity_distance(z))**2
    lumin = lumin.to(u.erg/u.s).value
    
    return lumin


def calc_lumin_from_prlx(flux, parallax):
    """
    Calculate luminosity from flux and parallax.

    Args:
        mag (float): apparent magnitude
        distance (float): distance in Mpc

    Returns:
        float: luminosity in erg/s
    """

    # parallax given im mas, distance is inversed parallax
    distance = (1000 / parallax) * u.pc

    lumin = 4 * np.pi * flux * (u.erg/u.s/u.cm**2) * distance**2
    lumin = lumin.to(u.erg/u.s).value
    
    return lumin


def G08_best_fit_soft(
        f: np.ndarray,
        f_ref: float = 1e-14,
        beta1: float = -1.58, beta2: float = -2.5,
        f_b: float = 10**(-0.04) * 1e-14,
        K: float = 1.51e16
        ) -> np.ndarray:
    
    """
    G08_best_fit_soft: best fit logNlogS from the paper of Georgakakis 2008 for
    soft 0.5-2 keV band. Those source counts are dominated by AGN

    Args:
        f (np.ndarray): flux
        f_ref (float, optional): parameter of logNlogS fitting.
        beta1 (float, optional): parameter of logNlogS fitting.
        beta2 (float, optional): parameter of logNlogS fitting.
        f_b (float, optional): parameter of logNlogS fitting.
        K (float, optional): parameter of logNlogS fitting.

    Returns:
        np.ndarray: logNlogS at given flux
    """

    Kprime = K * (f_b / f_ref) ** (beta1 - beta2)

    N = np.piecewise(
            f,
            [f < f_b, f >= f_b],
            [lambda f: K * (f_ref / (1 + beta1)) * ((f_b / f_ref)**(1 + beta1) -\
                (f / f_ref)**(1 + beta1)) - Kprime * f_ref / (1 + beta2)*(f_b / f_ref)**(1 + beta2),
            lambda f: -Kprime * f_ref / (1 + beta2) * (f / f_ref)**(1 + beta2)]
            )

    return N

def inclined_edge(x):
    """
    inclined edge of the tirangle area.
    """

    y = 1 + (1 - 0.5) / (3.5 - 1.4) * (x + 1.4)

    return y


def extragal_classifier(row):
    """
    Classify sources as extragalactic or not.

    Args:
        row (pd.Series): row of a dataframe

    Returns:
        Bool: True if extragalactic, False if not
    """    

    x='desi_rel_dered_lg(Fx/Fo_z_corr)'
    
    if row['class_final'] in ['GALACTIC (non-Gaia)', 'GALACTIC', 'STAR']:
        return False
    elif row['class_final'] in ['GALAXY', 'QSO']:
        return True
    elif row['class_final'] == 'UNKNOWN':
        if np.isnan(row[x]):
            return np.nan
        elif row['desi_type'] != 'PSF':
            return True
        elif row[x] > -1.4:
            return True
        else:
            return False


def validation_classifier(row):
    """
    Classify objects in the validation set.

    Args:
        row (pd.Series): row of the validation dataframe
    """

    x='desi_rel_dered_lg(Fx/Fo_z_corr)'

    if row['desi_type'] != 'PSF':
        return 'extragalactic'
    elif row[x] > -1.4:
        return 'extragalactic'
    else:
        return 'galactic'


def inside_beak(row):
    """
    Check if the point is inside the beak.
    """

    x = 'desi_rel_dered_lg(Fx/Fo_z_corr)'
    y = 'desi_rel_dered_r_z'

    if row[x] < -1.4 and row[y] > 0.5 and row[y] < inclined_edge(row[x]):
        return True
    else:
        return False
    

def plot_decision_scatter(df, ax=None, wise=False):
    """
    Plot scatter plot and decision boundary.

    Args:
        df (pd.DataFrame): dataframe with columns 'desi_rel_dered_lg(Fx/Fo_z_corr)'
                           and 'desi_rel_dered_r_z'
        ax (): axis to plot on

    Returns:
        ax: axis with the plot
    """

    if wise:
        x='desi_rel_dered_lg(Fx/Fo_z_corr)'

        y='desi_vega_mag_w1_w2'
        ylabel = 'color (w1 - w2)'

        xlims = np.array([-3.8, 2])
        ylims = [-0.3, 1.5]

        order = [3, 2, 0, 1]

        ax.axhline(0.05, color='k', linestyle='--', zorder=0)
    
    else:
        xlims = np.array([-3.8, 2])
        ylims = [-.5, 2.7]

        x='desi_rel_dered_lg(Fx/Fo_z_corr)'
        y='desi_rel_dered_r_z'

        ylabel = 'color (r - z)'

        order = [3, 2, 0, 1]

        plt.plot([-1.4, -3.5, -1.4], [1, 0.5, 0.5], c='gray', lw=1.5, ls='--')
        plt.axvline(-1.4, c='k', lw=2)

    print('Всего объектов:', len(df))
    print()

    unknown_df = df[df['class_final']=='UNKNOWN']
    galaxy_df = df[df['class_final']=='GALAXY']
    qso_df = df[df['class_final']=='QSO']
    galactic_df = df[df['class_final']=='GALACTIC']
    not_gaia_galactic_df = df.query('class_final in ["GALACTIC (non-Gaia)", "STAR"]')
    extended_df = df.query('desi_extended == True')

    # Scatterplots
    sns.scatterplot(
        data=unknown_df, x=x, y=y, ax=ax, s=20, color='gray', zorder=1,
        alpha=.3, label='UNKNOWN'
        )
    sns.scatterplot(
        data=extended_df, x=x, y=y, ax=ax, s=37, color='none', zorder=2,
        edgecolor='k', linewidth=.4, alpha=1, label='DESI LIS extended'
        )
    sns.scatterplot(
        data=qso_df, x=x, y=y, ax=ax, s=20, color='blue', zorder=3,
        label='EXTRAGAL (QSO)'
    )
    sns.scatterplot(
        data=galactic_df, x=x, y=y, ax=ax, s=20, color='r', zorder=10,
        label='GALACTIC'
        )
    sns.scatterplot(
        data=galaxy_df, x=x, y=y, ax=ax, s=20, color='lime', label='EXTRAGAL (GALAXY)', zorder=5
        )
    sns.scatterplot(
        data=not_gaia_galactic_df, x=x, y=y, ax=ax, s=30, color='none',
        edgecolor='r', linewidth=2, zorder=6, label='GALACTIC (non-Gaia)'
        )

    # ax.set_xlim(*xlims)
    # ax.set_ylim(*ylims)

    handles, labels = plt.gca().get_legend_handles_labels()
    lgnd = ax.legend(
        [handles[idx] for idx in order], [labels[idx] for idx in order],
        frameon=True, loc='upper right', fontsize=14
    )
    for handle in lgnd.legendHandles:
        handle._sizes = [70]

    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_xlabel(r'$\lg(F_x/F_o)$', fontsize=22)

    ax.tick_params(axis='both', which='major', labelsize=18)

    return ax

def assign_final_class(row):
    '''
    Assign final class based on the following priority:
    '''
    if not pd.isna(row['class_GAIA_class']):
        return row['class_GAIA_class']
    elif not pd.isna(row['class_SDSS_class']):
        return row['class_SDSS_class']
    elif not pd.isna(row['class_MILQ_class']):
        return row['class_MILQ_class']
    elif not pd.isna(row['class_SIMBAD_class']):
        return row['class_SIMBAD_class']
    else:
        return 'UNKNOWN'
    
def assign_class_source(row):
    '''
    Assign source of the final class
    '''

    if not pd.isna(row['class_GAIA_class']):
        return 'GAIA'
    elif not pd.isna(row['class_SDSS_class']):
        return 'SDSS'
    elif not pd.isna(row['class_MILQ_class']):
        return 'MILQ'
    elif not pd.isna(row['class_SIMBAD_class']):
        return 'SIMBAD'
    else:
        return np.nan

def assign_final_redshift(row):
    '''
    Assign final redshift based on the following priority:
    '''
    if row['class_final'] != 'GAIA':
        if not pd.isna(row.SDSS_zsp):
            return row.SDSS_zsp
        elif not pd.isna(row.MILQ_Z_spec):
            return row.MILQ_Z_spec
        elif not pd.isna(row.SIMBAD_z_rel):
            return row.SIMBAD_z_rel
        else:
            return np.nan
    
def assign_redshift_source(row):
    '''
    Assign final redshift based on the following priority:
    '''

    if not pd.isna(row.SDSS_zsp):
        return 'SDSS'
    elif not pd.isna(row.MILQ_Z_spec):
        return 'MILQ'
    elif not pd.isna(row.SIMBAD_z_rel):
        return 'SIMBAD'
    else:
        return np.nan
    

def assign_class_source_index(row):
    '''
    Assign source name from the additional catalogues
    '''
    if row['class_source'] == 'GAIA':
        return row['GAIA_source_id']
    elif row['class_source'] == 'SDSS':
        return row['SDSS_NAME']
    elif row['class_source'] == 'MILQ':
        return row['MILQ_NAME']
    elif row['class_source'] == 'SIMBAD':
        return row['MAIN_ID']
    else:
        return np.nan