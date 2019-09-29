import numpy as np
import pandas as pd
import sys


def binning_data(data, nbins, print_count =False):
    binned_data = pd.DataFrame()
    categorical_columns = ['gender', 'race', 'race_o', 'samerace', 'field', 'decision']
    continuous_columns = [col for col in data.columns if col not in categorical_columns]

    range_min = {}
    range_max = {}

    # Initialize ranges from pdf file
    range_min['age'], range_max['age'] = 18, 58
    range_min['age_o'], range_max['age_o'] = 18, 58
    range_min['importance_same_race'], range_max['importance_same_race'] = 0, 10
    range_min['importance_same_religion'], range_max['importance_same_religion'] = 0, 10

    range_min['pref_o_attractive'], range_max['pref_o_attractive'] = 0, 1
    range_min['pref_o_sincere'], range_max['pref_o_sincere'] = 0, 1
    range_min['pref_o_intelligence'], range_max['pref_o_intelligence'] = 0, 1
    range_min['pref_o_funny'], range_max['pref_o_funny'] = 0, 1
    range_min['pref_o_ambitious'], range_max['pref_o_ambitious'] = 0, 1
    range_min['pref_o_shared_interests'], range_max['pref_o_shared_interests'] = 0, 1

    range_min['attractive_important'], range_max['attractive_important'] = 0, 1
    range_min['sincere_important'], range_max['sincere_important'] = 0, 1
    range_min['intelligence_important'], range_max['intelligence_important'] = 0, 1
    range_min['funny_important'], range_max['funny_important'] = 0, 1
    range_min['ambition_important'], range_max['ambition_important'] = 0, 1
    range_min['shared_interests_important'], range_max['shared_interests_important'] = 0, 1

    range_min['attractive'], range_max['attractive'] = 0, 10
    range_min['sincere'], range_max['sincere'] = 0, 10
    range_min['intelligence'], range_max['intelligence'] = 0, 10
    range_min['funny'], range_max['funny'] = 0, 10
    range_min['ambition'], range_max['ambition'] = 0, 10

    range_min['attractive_partner'], range_max['attractive_partner'] = 0, 10
    range_min['sincere_partner'], range_max['sincere_partner'] = 0, 10
    range_min['intelligence_parter'], range_max['intelligence_parter'] = 0, 10
    range_min['funny_partner'], range_max['funny_partner'] = 0, 10
    range_min['ambition_partner'], range_max['ambition_partner'] = 0, 10
    range_min['shared_interests_partner'], range_max['shared_interests_partner'] = 0, 10

    range_min['sports'], range_max['sports'] = 0, 10
    range_min['tvsports'], range_max['tvsports'] = 0, 10
    range_min['exercise'], range_max['exercise'] = 0, 10
    range_min['dining'], range_max['dining'] = 0, 10
    range_min['museums'], range_max['museums'] = 0, 10
    range_min['art'], range_max['art'] = 0, 10
    range_min['hiking'], range_max['hiking'] = 0, 10
    range_min['gaming'], range_max['gaming'] = 0, 10
    range_min['clubbing'], range_max['clubbing'] = 0, 10
    range_min['reading'], range_max['reading'] = 0, 10
    range_min['tv'], range_max['tv'] = 0, 10
    range_min['theater'], range_max['theater'] = 0, 10
    range_min['movies'], range_max['movies'] = 0, 10
    range_min['concerts'], range_max['concerts'] = 0, 10
    range_min['music'], range_max['music'] = 0, 10
    range_min['shopping'], range_max['shopping'] = 0, 10
    range_min['yoga'], range_max['yoga'] = 0, 10

    range_min['interests_correlate'], range_max['interests_correlate'] = -1, 1
    range_min['expected_happy_with_sd_people'], range_max['expected_happy_with_sd_people'] = 0, 10
    range_min['like'], range_max['like'] = 0, 10

    # Fit values within range
    for col in continuous_columns:
        min_col = range_min[col]
        max_col = range_max[col]
        data[col] = [min_col if c < min_col else c for c in data[col]]
        data[col] = [max_col if c > max_col else c for c in data[col]]

    # Create bins for continuous columns and keep categorical columns
    for col in data.columns:
        if col in continuous_columns:
            min_col = range_min[col]
            max_col = range_max[col]
            bin_size = (max_col - min_col) / nbins
            bins = np.arange(min_col, max_col + bin_size, bin_size)
            binned_data[col] = pd.cut(data[col], bins, include_lowest=True)
            binned_data[col] = [str(c) for c in binned_data[col]]
            if print_count:
                print(f" {col}: {binned_data[col].value_counts(sort=False).values}")
        else:
            binned_data[col] = data[col]
    return binned_data


if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1])
    data_binned = binning_data(data, 5, True)
    data_binned.to_csv(index=False, path_or_buf=sys.argv[2])

