"""Helpers for various astronomical data products.
"""

import numpy as np

gdr2_source_meta_dtype = np.dtype([
    ('filename',        'U255'),
    ('filesize',        'i8'),
    ('source_id_start', 'i8'),
    ('source_id_end',   'i8'),
    ('sources',         'i8')
])

gdr2_csv_dtype = np.dtype([
    ('solution_id',                       'i8'),
    ('designation',                       'S30'),
    ('source_id',                         'i8'),
    ('random_index',                      'i8'),
    ('ref_epoch',                         'f8'),
    ('ra',                                'f8'),
    ('ra_error',                          'f8'),
    ('dec',                               'f8'),
    ('dec_error',                         'f8'),
    ('parallax',                          'f8'),
    ('parallax_error',                    'f8'),
    ('parallax_over_error',               'f8'),
    ('pmra',                              'f8'),
    ('pmra_error',                        'f8'),
    ('pmdec',                             'f8'),
    ('pmdec_error',                       'f8'),
    ('ra_dec_corr',                       'f4'),
    ('ra_parallax_corr',                  'f4'),
    ('ra_pmra_corr',                      'f4'),
    ('ra_pmdec_corr',                     'f4'),
    ('dec_parallax_corr',                 'f4'),
    ('dec_pmra_corr',                     'f4'),
    ('dec_pmdec_corr',                    'f4'),
    ('parallax_pmra_corr',                'f4'),
    ('parallax_pmdec_corr',               'f4'),
    ('pmra_pmdec_corr',                   'f4'),
    ('astrometric_n_obs_al',              'i4'),
    ('astrometric_n_obs_ac',              'i4'),
    ('astrometric_n_good_obs_al',         'i4'),
    ('astrometric_n_bad_obs_al',          'i4'),
    ('astrometric_gof_al',                'f4'),
    ('astrometric_chi2_al',               'f4'),
    ('astrometric_excess_noise',          'f8'),
    ('astrometric_excess_noise_sig',      'f8'),
    ('astrometric_params_solved',         'int8'),
    ('astrometric_primary_flag',          'bool'),
    ('astrometric_weight_al',             'f4'),
    ('astrometric_pseudo_colour',         'f8'),
    ('astrometric_pseudo_colour_error',   'f8'),
    ('mean_varpi_factor_al',              'f4'),
    ('astrometric_matched_observations',  'i2'),
    ('visibility_periods_used',           'i2'),
    ('astrometric_sigma5d_max',           'f4'),
    ('frame_rotator_object_type',         'i4'),
    ('matched_observations',              'i2'),
    ('duplicated_source',                 'bool'),
    ('phot_g_n_obs',                      'i4'),
    ('phot_g_mean_flux',                  'f8'),
    ('phot_g_mean_flux_error',            'f8'),
    ('phot_g_mean_flux_over_error',       'f4'),
    ('phot_g_mean_mag',                   'f4'),
    ('phot_bp_n_obs',                     'i4'),
    ('phot_bp_mean_flux',                 'f8'),
    ('phot_bp_mean_flux_error',           'f8'),
    ('phot_bp_mean_flux_over_error',      'f4'),
    ('phot_bp_mean_mag',                  'f4'),
    ('phot_rp_n_obs',                     'i4'),
    ('phot_rp_mean_flux',                 'f8'),
    ('phot_rp_mean_flux_error',           'f8'),
    ('phot_rp_mean_flux_over_error',      'f4'),
    ('phot_rp_mean_mag',                  'f4'),
    ('phot_bp_rp_excess_factor',          'f4'),
    ('phot_proc_mode',                    'int8'),
    ('bp_rp',                             'f4'),
    ('bp_g',                              'f4'),
    ('g_rp',                              'f4'),
    ('radial_velocity',                   'f8'),
    ('radial_velocity_error',             'f8'),
    ('rv_nb_transits',                    'i4'),
    ('rv_template_teff',                  'f4'),
    ('rv_template_logg',                  'f4'),
    ('rv_template_fe_h',                  'f4'),
    ('phot_variable_flag',                'S15'),
    ('l',                                 'f8'),
    ('b',                                 'f8'),
    ('ecl_lon',                           'f8'),
    ('ecl_lat',                           'f8'),
    ('priam_flags',                       'i8'),
    ('teff_val',                          'f4'),
    ('teff_percentile_lower',             'f4'),
    ('teff_percentile_upper',             'f4'),
    ('a_g_val',                           'f4'),
    ('a_g_percentile_lower',              'f4'),
    ('a_g_percentile_upper',              'f4'),
    ('e_bp_min_rp_val',                   'f4'),
    ('e_bp_min_rp_percentile_lower',      'f4'),
    ('e_bp_min_rp_percentile_upper',      'f4'),
    ('flame_flags',                       'i8'),
    ('radius_val',                        'f4'),
    ('radius_percentile_lower',           'f4'),
    ('radius_percentile_upper',           'f4'),
    ('lum_val',                           'f4'),
    ('lum_percentile_lower',              'f4'),
    ('lum_percentile_upper',              'f4')
])
gdr2_csv_dtype_unicode = np.dtype([
    ('solution_id',                       'i8'),
    ('designation',                       'U30'),
    ('source_id',                         'i8'),
    ('random_index',                      'i8'),
    ('ref_epoch',                         'f8'),
    ('ra',                                'f8'),
    ('ra_error',                          'f8'),
    ('dec',                               'f8'),
    ('dec_error',                         'f8'),
    ('parallax',                          'f8'),
    ('parallax_error',                    'f8'),
    ('parallax_over_error',               'f8'),
    ('pmra',                              'f8'),
    ('pmra_error',                        'f8'),
    ('pmdec',                             'f8'),
    ('pmdec_error',                       'f8'),
    ('ra_dec_corr',                       'f4'),
    ('ra_parallax_corr',                  'f4'),
    ('ra_pmra_corr',                      'f4'),
    ('ra_pmdec_corr',                     'f4'),
    ('dec_parallax_corr',                 'f4'),
    ('dec_pmra_corr',                     'f4'),
    ('dec_pmdec_corr',                    'f4'),
    ('parallax_pmra_corr',                'f4'),
    ('parallax_pmdec_corr',               'f4'),
    ('pmra_pmdec_corr',                   'f4'),
    ('astrometric_n_obs_al',              'i4'),
    ('astrometric_n_obs_ac',              'i4'),
    ('astrometric_n_good_obs_al',         'i4'),
    ('astrometric_n_bad_obs_al',          'i4'),
    ('astrometric_gof_al',                'f4'),
    ('astrometric_chi2_al',               'f4'),
    ('astrometric_excess_noise',          'f8'),
    ('astrometric_excess_noise_sig',      'f8'),
    ('astrometric_params_solved',         'int8'),
    ('astrometric_primary_flag',          'bool'),
    ('astrometric_weight_al',             'f4'),
    ('astrometric_pseudo_colour',         'f8'),
    ('astrometric_pseudo_colour_error',   'f8'),
    ('mean_varpi_factor_al',              'f4'),
    ('astrometric_matched_observations',  'i2'),
    ('visibility_periods_used',           'i2'),
    ('astrometric_sigma5d_max',           'f4'),
    ('frame_rotator_object_type',         'i4'),
    ('matched_observations',              'i2'),
    ('duplicated_source',                 'bool'),
    ('phot_g_n_obs',                      'i4'),
    ('phot_g_mean_flux',                  'f8'),
    ('phot_g_mean_flux_error',            'f8'),
    ('phot_g_mean_flux_over_error',       'f4'),
    ('phot_g_mean_mag',                   'f4'),
    ('phot_bp_n_obs',                     'i4'),
    ('phot_bp_mean_flux',                 'f8'),
    ('phot_bp_mean_flux_error',           'f8'),
    ('phot_bp_mean_flux_over_error',      'f4'),
    ('phot_bp_mean_mag',                  'f4'),
    ('phot_rp_n_obs',                     'i4'),
    ('phot_rp_mean_flux',                 'f8'),
    ('phot_rp_mean_flux_error',           'f8'),
    ('phot_rp_mean_flux_over_error',      'f4'),
    ('phot_rp_mean_mag',                  'f4'),
    ('phot_bp_rp_excess_factor',          'f4'),
    ('phot_proc_mode',                    'int8'),
    ('bp_rp',                             'f4'),
    ('bp_g',                              'f4'),
    ('g_rp',                              'f4'),
    ('radial_velocity',                   'f8'),
    ('radial_velocity_error',             'f8'),
    ('rv_nb_transits',                    'i4'),
    ('rv_template_teff',                  'f4'),
    ('rv_template_logg',                  'f4'),
    ('rv_template_fe_h',                  'f4'),
    ('phot_variable_flag',                'U15'),
    ('l',                                 'f8'),
    ('b',                                 'f8'),
    ('ecl_lon',                           'f8'),
    ('ecl_lat',                           'f8'),
    ('priam_flags',                       'i8'),
    ('teff_val',                          'f4'),
    ('teff_percentile_lower',             'f4'),
    ('teff_percentile_upper',             'f4'),
    ('a_g_val',                           'f4'),
    ('a_g_percentile_lower',              'f4'),
    ('a_g_percentile_upper',              'f4'),
    ('e_bp_min_rp_val',                   'f4'),
    ('e_bp_min_rp_percentile_lower',      'f4'),
    ('e_bp_min_rp_percentile_upper',      'f4'),
    ('flame_flags',                       'i8'),
    ('radius_val',                        'f4'),
    ('radius_percentile_lower',           'f4'),
    ('radius_percentile_upper',           'f4'),
    ('lum_val',                           'f4'),
    ('lum_percentile_lower',              'f4'),
    ('lum_percentile_upper',              'f4')
])