import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from openai import OpenAI

def load_data(filepath):
    """ Load data from a CSV file. """
    return pd.read_csv(filepath)

def simplify_house_category(df):
    # Calculate the mean year for 'Year Built' and 'Year Remod/Add'
    year_built_mean = df['YearBuilt'].mean()
    year_remod_mean = df['YearRemodAdd'].mean()

    # Define binary categories for 'Year Built' and 'Year Remod/Add' based on the mean
    df['YearBuiltCategory'] = pd.cut(df['YearBuilt'], bins=[df['YearBuilt'].min()-1, year_built_mean, df['YearBuilt'].max()], 
                                     labels=[f'Old', f'New'])
    df['YearRemodCategory'] = pd.cut(df['YearRemodAdd'], bins=[df['YearRemodAdd'].min()-1, year_remod_mean, df['YearRemodAdd'].max()], 
                                     labels=[f'NotRecentlyRemodeled', f'RecentlyRemodeled'])

    # Simplify 'House Style' into 'Single-Level' and 'Multi-Level'
    def simplify_style(style):
        if style in ['1Story', 'SFoyer', 'SLvl']:
            return 'Single-Level'
        else:
            return 'Multi-Level'
        
    # Simplify building types by combining townhouse types
    def simplify_building_type(bldg_type):
        if bldg_type in ['TwnhsE', 'TwnhsI']:
            return 'Townhouse'
        elif bldg_type == '2FmCon':
            return 'Two-Family'
        elif bldg_type == 'Duplx':
            return 'Duplex'
        else:
            return 'Single-Family'
        
    # Apply building type simplification
    df['BldgType'] = df['BldgType'].apply(simplify_building_type)

    # Apply the 'simplify_style' function to the 'HouseStyle' column
    df['SimplifiedStyle'] = df['HouseStyle'].apply(simplify_style)

    # Create a new 'House Category' variable based on simplified 'House Style', 'Bldg Type', and the new year categories
    df['HouseCategory'] = df.apply(lambda row: f"{row['SimplifiedStyle']}-{row['BldgType']}-{row['YearBuiltCategory']}-{row['YearRemodCategory']}", axis=1)

    # Print the unique values of the new 'House Category' variable
    print(df['HouseCategory'].unique())

    # Optionally, drop the columns that are no longer needed if they won't be used further
    df.drop(['HouseStyle', 'BldgType', 'YearBuilt', 'YearRemodAdd', 'YearBuiltCategory', 'YearRemodCategory', 'SimplifiedStyle'], axis=1, inplace=True)
    return df

def preprocess_data(df):
    """ Preprocess the data: handle missing values, encode categorical variables, normalize data. """
    # Drop Id
    df = df.drop("Id", axis=1)

    # Drop columns with more than 25% missing values
    missing_values = df.isnull().mean()
    columns_to_drop = missing_values[missing_values > 0.25].index
    df = df.drop(columns_to_drop, axis=1)

    # Drop numerical columns with more than 90% zeros
    zeros = (df == 0).mean()
    columns_to_drop = zeros[zeros > 0.90].index
    df = df.drop(columns_to_drop, axis=1)

    # Simplify 'House Category'
    df = simplify_house_category(df)

    # Separate target variable if they are present in the dataframe
    if 'SalePrice' in df.columns:
        targets = df[['SalePrice', 'HouseCategory']]
        df = df.drop(['SalePrice', 'HouseCategory'], axis=1)
        train_data = True
    else:
        # Must be preprocessing the test data
        df = df.drop('HouseCategory', axis=1)
        train_data = False

    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Print length of numerical and categorical columns
    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    # Encoding categorical data
    encoder = OrdinalEncoder()
    df_encoded = df.copy()
    df_encoded[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    # Imputation
    imputer = KNNImputer(n_neighbors=7)
    df_encoded[numerical_cols] = imputer.fit_transform(df_encoded[numerical_cols])
    df_encoded[categorical_cols] = imputer.fit_transform(df_encoded[categorical_cols])

    # Decode back to original categories
    df[categorical_cols] = encoder.inverse_transform(df_encoded[categorical_cols])
    df[numerical_cols] = df_encoded[numerical_cols]

    # Normalize numerical colums with Min-Max scaling (sklearn MinMaxScaler)
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Use embeddings for categorical columns
    client = OpenAI()

    def get_embedding(text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model, dimensions=35).data[0].embedding

    # Concatenate all categorical columns and get a single embedding for the concatenated string, then split into separate columns
    concatenated_cols = df[categorical_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    embeddings = concatenated_cols.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
    embeddings_df = pd.DataFrame(embeddings.tolist(), columns=[f'embedding_{i}' for i in range(35)])
    scaler = MinMaxScaler()
    embeddings_df_scaled = pd.DataFrame(scaler.fit_transform(embeddings_df), columns=embeddings_df.columns)
    df = pd.concat([df, embeddings_df_scaled], axis=1)
    print("Embedding and scaling for concatenated categorical columns completed.")

    # Drop the original categorical columns
    df = df.drop(categorical_cols, axis=1)

    if train_data:
        # Preprocess the target variable
        encoder = OrdinalEncoder()
        targets['HouseCategory'] = encoder.fit_transform(targets['HouseCategory'].values.reshape(-1, 1))

        # Append the target variable to the dataframe
        df['SalePrice'] = targets['SalePrice']
        df['HouseCategory'] = targets['HouseCategory']

    return df

if __name__ == "__main__":
    data = load_data("../data/raw/train.csv")
    processed_data = preprocess_data(data)
    processed_data.to_csv("../data/processed/processed_train.csv", index=False)
