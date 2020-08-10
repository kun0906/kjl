import numpy as np
import pandas as pd
from pandas import ExcelWriter


def diff_xlsx(xlsx_file1='', xlsx_file2=''):
    """
        https://kanoki.org/2019/02/26/compare-two-excel-files-for-difference-using-python/
        
    """
    # xls = pd.ExcelWriter(xlsx_file1, engine='xlsxwriter')
    xls = pd.ExcelFile(xlsx_file1)
    print(f'xls.sheet_names:', xls.sheet_names)
    output_file = f'{xlsx_file1}-diff.xlsx'
    with ExcelWriter(output_file) as writer:
        for i, sheet_name in enumerate(xls.sheet_names):
            print(i, sheet_name)

            df1 = pd.read_excel(xlsx_file1, sheet_name)
            df2 = pd.read_excel(xlsx_file2, sheet_name)

            # replace nan of df
            df1 = df1.replace(np.nan, '')
            df2 = df2.replace(np.nan, '')

            df1.equals(df2)

            comparison_values = df1.values == df2.values
            print(comparison_values)

            rows, cols = np.where(comparison_values == False)

            for item in zip(rows, cols):
                df1.iloc[item[0], item[1]] = '{} --> {}'.format(df1.iloc[item[0], item[1]], df2.iloc[item[0], item[1]])

            # Generate dataframe from list and write to xlsx.

            pd.DataFrame(df1.values).to_excel(writer, sheet_name=sheet_name, index=False, header=True)

        writer.save()

    return output_file


if __name__ == '__main__':
    xlsx_file1 = 'out/data_kjl/report/KJL-OCSVM-GMM-gs_True-20200725.xlsx'
    xlsx_file2 = 'out/data_kjl/report/KJL-OCSVM-GMM-gs_True-20200724.xlsx'
    diff_xlsx(xlsx_file1, xlsx_file2)
