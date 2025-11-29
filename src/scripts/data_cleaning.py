import os
import pandas as pd
from src.utils.utils import load_data, save_data

print("Reading Data")
df_liar_train = load_data("data/datasets/LIARFakeNewsDataset/raw/train.tsv")
df_liar_valid = load_data("data/datasets/LIARFakeNewsDataset/raw/valid.tsv")
df_liar_test = load_data("data/datasets/LIARFakeNewsDataset/raw/test.tsv")

df_isot_fake = load_data("data/datasets/ISOT_FakeNewsDataset/News_Dataset/raw/Fake.csv")
df_isot_true = load_data("data/datasets/ISOT_FakeNewsDataset/News_Dataset/raw/True.csv")

df_fakenewsnet_buzzfeed_fake = load_data("data/datasets/FakeNewsNet/raw/BuzzFeed_fake_news_content.csv")
df_fakenewsnet_buzzfeed_true = load_data("data/datasets/FakeNewsNet/raw/BuzzFeed_real_news_content.csv")
df_fakenewsnet_politifact_fake = load_data("data/datasets/FakeNewsNet/raw/PolitiFact_fake_news_content.csv")
df_fakenewsnet_politifact_true = load_data("data/datasets/FakeNewsNet/raw/PolitiFact_real_news_content.csv")

df_liar_train.columns = ["ID", "label", "statement", "subjects", "speaker", "speaker_job", "state", "party", 'barely true counts', 'false counts', 'half true counts', 'mostly true counts', 'pants on fire counts', "venue"]
df_liar_valid.columns = ["ID", "label", "statement", "subjects", "speaker", "speaker_job", "state", "party", 'barely true counts', 'false counts', 'half true counts', 'mostly true counts', 'pants on fire counts', "venue"]
df_liar_test.columns = ["ID", "label", "statement", "subjects", "speaker", "speaker_job", "state", "party", 'barely true counts', 'false counts', 'half true counts', 'mostly true counts', 'pants on fire counts', "venue"]

liar_columns_to_drop = ["ID", "subjects", "speaker", "speaker_job", "state", "party", 'barely true counts', 'false counts', 'half true counts', 'mostly true counts', 'pants on fire counts', "venue"]
isot_column_to_drop = ["date", "subject"]
fakenews_columns_to_drop = ["id", "url", "top_img", "authors", "source", "publish_date", "movies", "images", "canonical_link", "meta_data"]
print("Done Reading")

'''Droping unnecesary data'''
df_liar_train.drop(liar_columns_to_drop, axis=1, inplace=True)
df_liar_valid.drop(liar_columns_to_drop, axis=1, inplace=True)
df_liar_test.drop(liar_columns_to_drop, axis=1, inplace=True)

df_fakenewsnet_buzzfeed_fake.drop(fakenews_columns_to_drop, axis=1, inplace=True)
df_fakenewsnet_buzzfeed_true.drop(fakenews_columns_to_drop, axis=1, inplace=True)
df_fakenewsnet_politifact_fake.drop(fakenews_columns_to_drop, axis=1, inplace=True)
df_fakenewsnet_politifact_true.drop(fakenews_columns_to_drop, axis=1, inplace=True)

df_isot_fake.drop(isot_column_to_drop, axis=1, inplace=True)
df_isot_true.drop(isot_column_to_drop, axis=1, inplace=True)

'''Create new 'label' column'''
df_fakenewsnet_buzzfeed_fake["label"] = 0
df_fakenewsnet_buzzfeed_true["label"] = 1
df_fakenewsnet_politifact_fake["label"] = 0
df_fakenewsnet_politifact_true["label"] = 1

df_isot_fake["label"] = 0
df_isot_true["label"] = 1

'''Combining the datasets'''
df_liar_combined = pd.concat([df_liar_train,df_liar_valid, df_liar_test],axis=0).sample(frac = 1, random_state = 42).reset_index(drop = True)
df_isot_combined = pd.concat([df_isot_fake,df_isot_true],axis=0).sample(frac = 1, random_state = 42).reset_index(drop = True)
df_fakenewsnet_combined = pd.concat([df_fakenewsnet_buzzfeed_fake,df_fakenewsnet_buzzfeed_true, df_fakenewsnet_politifact_fake, df_fakenewsnet_politifact_true],axis=0).sample(frac = 1, random_state = 42).reset_index(drop = True)


liar_df_label_mapping = {
    'true': 1,
    'mostly-true': 1,
    'half-true': 0,
    'barely-true': 0,
    'false': 0,
    'pants-fire': 0
}
df_liar_filtered = df_liar_combined[df_liar_combined['label'].isin(liar_df_label_mapping.keys())]
df_liar_filtered['label'] = df_liar_filtered['label'].str.lower().map(liar_df_label_mapping)

'''Droping Duplicates'''
df_liar_filtered = df_liar_filtered.drop_duplicates()
df_liar_filtered.reset_index(drop=True, inplace=True)

df_isot_filtered = df_isot_combined.drop_duplicates()
df_isot_filtered.reset_index(drop=True, inplace=True)
df_isot_filtered = df_isot_filtered.copy()
df_isot_filtered["statement"] = df_isot_filtered["title"] + " " + df_isot_filtered["text"]
df_isot_filtered.drop(["title", "text"], axis=1, inplace=True)

df_fakenewsnet_filtered = df_fakenewsnet_combined.drop_duplicates()
df_fakenewsnet_filtered.reset_index(drop=True, inplace=True)
df_fakenewsnet_filtered = df_fakenewsnet_filtered.copy()
df_fakenewsnet_filtered["statement"] = df_fakenewsnet_filtered["title"] + " " + df_fakenewsnet_filtered["text"]
df_fakenewsnet_filtered.drop(["title", "text"], axis=1, inplace=True)

save_data(df_liar_filtered, 'data/datasets/LIARFakeNewsDataset/processed', 'liar_data.csv')
save_data(df_isot_filtered, 'data/datasets/ISOT_FakeNewsDataset/News_Dataset/processed', 'isot.csv')
save_data(df_fakenewsnet_filtered, 'data/datasets/FakeNewsNet/processed', 'fakenews.csv')

df_total_combined = pd.concat([df_liar_filtered, df_isot_filtered, df_fakenewsnet_filtered],axis=0).sample(frac = 1, random_state = 42).reset_index(drop = True)
save_data(df_total_combined, 'data/datasets/final_data', 'data.csv')
print("Done Again")