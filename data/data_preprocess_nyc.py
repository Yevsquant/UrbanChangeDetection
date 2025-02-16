import pandas as pd
import os

def preprocess(df, mask_path_1, mask_path_2, image_path_1, image_path_2, image_path_3):
    data_mask = df.copy()
    data_mask["min_panoid_image"] = df["min_panoid_mask"].str.replace('_mask', '')
    data_mask["max_panoid_image"] = df["max_panoid_mask"].str.replace('_mask', '')

    # Add full image mask path to the dataframe
    min_mask = []
    max_mask = []
    for i in range(len(data_mask)):
        current_min_path_1 = mask_path_1 + data_mask.iloc[i]['min_panoid_mask']
        current_min_path_2 = mask_path_2 + data_mask.iloc[i]['min_panoid_mask']
        if os.path.exists(current_min_path_1):
            min_mask.append(current_min_path_1)
        elif os.path.exists(current_min_path_2):
            min_mask.append(current_min_path_2)

        current_max_path_1 = mask_path_1 + data_mask.iloc[i]['max_panoid_mask']
        current_max_path_2 = mask_path_2 + data_mask.iloc[i]['max_panoid_mask']
        if os.path.exists(current_max_path_1):
            max_mask.append(current_max_path_1)
        elif os.path.exists(current_max_path_2):
            max_mask.append(current_max_path_2)

    data_mask["min_panoid_mask location"] = min_mask
    data_mask["max_panoid_mask location"] = max_mask

    # Add full image path to the dataframe
    min_image = []
    max_image = []
    for i in range(len(data_mask)):
        current_min_path_1 = image_path_1 + data_mask.iloc[i]['min_panoid_image']
        current_min_path_2 = image_path_2 + data_mask.iloc[i]['min_panoid_image']
        current_min_path_3 = image_path_3 + data_mask.iloc[i]['min_panoid_image']
        if os.path.exists(current_min_path_1):
            min_image.append(current_min_path_1)
        elif os.path.exists(current_min_path_2):
            min_image.append(current_min_path_2)
        elif os.path.exists(current_min_path_3):
            min_image.append(current_min_path_3)

        current_max_path_1 = image_path_1 + data_mask.iloc[i]['max_panoid_image']
        current_max_path_2 = image_path_2 + data_mask.iloc[i]['max_panoid_image']
        current_max_path_3 = image_path_3 + data_mask.iloc[i]['max_panoid_image']
        if os.path.exists(current_max_path_1):
            max_image.append(current_max_path_1)
        elif os.path.exists(current_max_path_2):
            max_image.append(current_max_path_2)
        elif os.path.exists(current_max_path_3):
            max_image.append(current_max_path_3)

    data_mask["min image location"] = min_image
    data_mask["max image location"] = max_image

    # Split the data into training and validation sets
    df_train = pd.DataFrame(columns=data_mask.columns)
    df_val = pd.DataFrame(columns=data_mask.columns)
    for index, row in data_mask.iterrows():
        if (index + 1) % 5 == 0:
            df_val.loc[len(df_val)] = row
        else:
            df_train.loc[len(df_train)] = row
    
    df_train["temp_id"] = range(0, len(df_train))
    df_val["temp_id"] = range(0, len(df_val))

    return df_train, df_val
