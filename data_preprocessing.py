from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

class FeatureExtractor:
    def __init__(self, fc_parameters=None):
        # Устанавливаем параметры извлечения признаков
        if fc_parameters is None:
            self.fc_parameters = {
                'sum_values': None,
                'agg_linear_trend': [
                    {"attr": "slope", "chunk_len": 10, "f_agg": "mean"},
                    {"attr": "intercept", "chunk_len": 10, "f_agg": "min"},
                    {"attr": "slope", "chunk_len": 10, "f_agg": "max"}
                ],
                'last_location_of_minimum': None,
                'mean': None,
                'approximate_entropy': [{"m": 2, "r": 0.9}],
                'fft_aggregated': [{"aggtype": "kurtosis"}],
                'variance_larger_than_standard_deviation': None,
                'fft_coefficient': [{"attr": "real", "coeff": 1}]
            }
        else:
            self.fc_parameters = fc_parameters

        # Словарь для переименования столбцов
        self.rename_dict = {
            'values__sum_values': 'sum_values',
            'values__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"': 'linear_trend_slope_mean_chunk_10',
            'values__last_location_of_minimum': 'last_location_of_min',
            'values__mean': 'mean_value',
            'values__approximate_entropy__m_2__r_0.9': 'approximate_entropy_m2_r0.9',
            'values__fft_aggregated__aggtype_"kurtosis"': 'fft_kurtosis',
            'values__variance_larger_than_standard_deviation': 'variance_larger_than_std',
            'values__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"': 'linear_trend_intercept_min_chunk_10',
            'values__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"max"': 'linear_trend_slope_max_chunk_10',
            'values__fft_coefficient__attr_"real"__coeff_1': 'fft_real_coeff_1'
        }

        # Заранее создаём список столбцов в нужном порядке
        self.ordered_columns = [
            'values__sum_values',
            'values__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"',
            'values__last_location_of_minimum',
            'values__mean',
            'values__approximate_entropy__m_2__r_0.9',
            'values__fft_aggregated__aggtype_"kurtosis"',
            'values__variance_larger_than_standard_deviation',
            'values__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"',
            'values__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"max"',
            'values__fft_coefficient__attr_"real"__coeff_1'
        ]

    def prepare_data(self, df):
        """Функция для подготовки данных"""
        df = df.explode(['dates', 'values']).reset_index(drop=True)
        df.fillna(0, inplace = True) 
        return df

    def extract_and_process_features(self, df):
        """Функция для извлечения и обработки признаков"""
        print(f'Идёт подготовка данных... (около 2 минут)')

        # Извлечение признаков
        features = extract_features(
            df,
            column_id='id',
            column_sort='dates',
            column_value='values',
            default_fc_parameters=self.fc_parameters,
            n_jobs=4
        )

        # Импутация пропущенных значений
        features = impute(features)

        # Приведение столбцов к нужному порядку перед переименованием
        existing_columns = [col for col in self.ordered_columns if col in features.columns]
        features = features[existing_columns]

        # Переименование столбцов в соответствии с заранее заданным словарем
        features.rename(columns=self.rename_dict, inplace=True)

        # Возвращение индексов в том же порядке, как в исходном DataFrame
        features = features.reindex(df['id'].drop_duplicates().values)

        print(f'Подготовка данных завершена')

        return features
