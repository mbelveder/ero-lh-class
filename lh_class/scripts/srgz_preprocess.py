#!/usr/bin/env python3

# This script provides a preprocessing procedure that adds useful x-ray
# features to the SRGz catalog (with p(z))

import pandas as pd

class_df = pd.read_pickle(
    f'{data_path}/paper_lh_z_upd_new_columns_df.gz_pkl',
    compression='gzip'
    )

fp = 'data/lhpv_03_23_sd01_a15_g14_srgz_CatA_XnX_model4_SQG_model5_v20221207'
full_srgz_df = pd.read_pickle(fp, compression='gzip')
full_srgz_df.srcname_fin = full_srgz_df.srcname_fin.str.decode('utf-8')
# SRGz catalog (only best counterparts)
srgz_df = full_srgz_df.query('srg_match_flag==1')

print(f'Всего источников SRGz: {len(srgz_df)}')

print(f'Всего источников nway: {len(class_df)}')