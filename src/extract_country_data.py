import pandas as pd
import quandl


def get_country_codes(country):
    df_codes = pd.read_csv("SGE_codes.csv")
    codes_list = []
    for code in df_codes['code'].items():
        if country in code[1]:
            codes_list.append(code[1])

    df = quandl.get(codes_list[0])
    df.rename(columns={'Value' : codes_list[0]}, inplace=True)


    for code in codes_list[1:]:
        df_temp = quandl.get(code)
        df_temp.rename(columns={'Value' : code}, inplace=True)
        df = pd.concat([df, df_temp], axis='columns')
    return df

if __name__ == "__main__":
    quandl.ApiConfig.api_key = ""
    country = "USA"
    country_codes = get_country_codes(country)

    # Chinese stock data:
    df = quandl.get_table('DY/SPA', ticker='600170')
    df.to_csv(f"Data_{country}.csv")
