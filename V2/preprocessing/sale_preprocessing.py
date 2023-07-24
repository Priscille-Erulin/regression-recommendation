import pandas as pd
import pickle
import numpy as np
import unidecode as udec

BADGE_FEATURES = ['artisanal', 'b_corporation', 'bio', 'biodegradable',
                  'cadeau_ideal', 'concept_original', 'durable',
                  'excellent_sur_yuka', 'exclusivite_choose',
                  'fabrication_a_la_demande', 'fait_main', 'gluten_free',
                  'iconique', 'inclusive', 'innovation', 'made_in_europe',
                  'made_in_france', 'madeinjapan', 'naturel', 'oeko_tex',
                  'premium', 'recyclable', 'saint_valentin', 'savoir_faire',
                  'seconde_main', 'socialement_engagee', 'serie_limitee',
                  'tendance', 'upcycling', 'vegan', 'vintage', 'zerodechet',
                  'eco_friendly']

CATEGORY_FEATURES = [None, 'Accessoires', 'Beauté', 'Bibliothèque',
                     'Bien-être', 'Bijoux', 'Buanderie', 'Chambre',
                     'Chaussant', 'Chaussures', 'Cuisine', 'Cures',
                     'Expériences', 'Hygiène', 'Kids', 'Lingerie',
                     'Maroquinerie', 'Outdoor', 'Prêt-à-porter',
                     'Salon', 'Soins', 'Sportswear']
mapping_categ = {category: index + 1 for index, category in enumerate(CATEGORY_FEATURES)}

LOG_COLUMNS = ['delta', 'followers', 'revenue', 'brand_appearance',
               'avg_price']

NUMERICAL_COLUMNS = ['log_delta', 'log_followers', 'conversion',
                     'log_revenue', 'log_brand_appearance', 'log_avg_price']

scaler = pickle.load(open('training_preparation/scaler_sales.sav', 'rb'))


def dummify_badges(badges):
    """
    Return dummies of badges according to the encoded list.
    """
    # formating of the badge names
    badges = udec.unidecode((badges)
                            .lower()
                            .replace(" ", "_")
                            .replace("-", "_")
                            )
    if badges is None:
        badges = ' '
    return [int(x in badges) for x in BADGE_FEATURES]


def tokenise_categories(df):
    token_df = df['category'].copy(deep=True)
    token_df = pd.DataFrame([mapping_categ[categ] 
                            if categ in CATEGORY_FEATURES 
                            else 0 
                            for categ in token_df], columns=['category_sale'])
    return token_df


def transform_to_log(df: pd.DataFrame, columns_to_log: list):
    """
    Adding columns using log of provided information.
    """
    logged_df = df.copy(deep=True)
    for column_name in columns_to_log:
        logged_df['log_' + column_name] = np.log(logged_df[column_name] + 1)
        logged_df = logged_df.drop(column_name, axis='columns')
    return logged_df


def scale(df: pd.DataFrame, columns_to_scale: list, scaler=scaler):
    """
    Return dataframe with the given columns scaled.
    """
    scaled_df = df.copy(deep=True)
    scaled_df[columns_to_scale] = scaler.transform(scaled_df[columns_to_scale])
    return scaled_df


def process_sale_features(df: pd.DataFrame):
    """
    Return a preprocessed df after dummification, log and scaling.
    """
    df = df.dropna().reset_index(drop=True)
    sale_ids = df['sale_id']

    dummies_category = tokenise_categories(df)
    dummies_badges = pd.DataFrame([dummify_badges(badges)
                                   for badges in df['badges']],
                                  columns=BADGE_FEATURES)

    num_output_df = df[LOG_COLUMNS + ['conversion']]
    num_output_df = transform_to_log(num_output_df, LOG_COLUMNS)
    num_output_df = scale(num_output_df, NUMERICAL_COLUMNS)

    concatenated_df = pd.concat([sale_ids, num_output_df,
                                 dummies_badges,
                                 dummies_category], axis=1)

    return concatenated_df
