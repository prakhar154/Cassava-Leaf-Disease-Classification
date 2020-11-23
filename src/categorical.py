import pandas as pd

class CategoricalFeatures:

	def __init__(self, df, categorical_features, encoding):
		self.df = df
		self.cat_features = categorical_features
		self.enc = encoding
		self.output_df = self.df.copy(deep=True)

	def _one_hot(self):
		self.output_df = pd.get_dummies(data=self.output_df, columns=self.cat_features)
		return self.output_df

if __name__ == '__main__':
	df = pd.read_csv('./input/train_folds.csv')
	cols = ['label']

	cat_feats = CategoricalFeatures(df, categorical_features=cols, encoding='ohe')
	train_ohe = cat_feats._one_hot()

	train_ohe.to_csv('./input/train_ohe.csv', index=False)
