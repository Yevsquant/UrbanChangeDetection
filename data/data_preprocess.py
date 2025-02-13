import pandas as pd

def preprocess(df):
    data = df.copy()
    data = data.drop(["min_year", "max_year", 'left mask sum', 'right mask sum', 'min_panoid', 'max_panoid'], axis=1)
    #data['cluster_id'] = data['cluster_id'].apply(convert_to_float)
    data['cluster_id'] = data['cluster_id'].astype(str)
    data['image'] = data['cluster_id'].astype(str) + "_" + data["time_period"] + ".jpg"
    data = data.drop(columns=["time_period", "cluster_id"])
    data["temp_id"] = range(0, len(data))
    data["label"].value_counts()
    data.to_csv('data.csv', index = False)
    data_mask = data.copy()

    """
    # downsampling
    df_majority = data_mask[data_mask.label == 0]
    df_minority = data_mask[data_mask.label == 1]

    majority_downsampled = df_majority.sample(n=len(df_minority), replace=True)

    df_balanced = pd.concat([df_minority, majority_downsampled])
    """

    """
    # the first quarter of the trainset
    df_balanced = data_mask.iloc[:len(data_mask)//4]
    """

    # upsampling
    df_train = pd.DataFrame(columns=data_mask.columns)
    df_val = pd.DataFrame(columns=data_mask.columns)

    for index, row in data_mask.iterrows():
        if (index + 1) % 5 == 0:
            df_val.loc[len(df_val)] = row
        else:
            df_train.loc[len(df_train)] = row

    df_majority = df_train[df_train.label == 0]
    df_minority = df_train[df_train.label == 1]

    minority_upsampled = df_minority.sample(n=len(df_majority), replace=True)

    df_train = pd.concat([df_majority, minority_upsampled])

    print(len(df_train), len(df_val))