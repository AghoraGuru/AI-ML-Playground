import gradio as gr
import pandas as pd
from pycaret.classification import setup, compare_models, get_leaderboard
import ydata_profiling as prof

class PycaretModel:
    def __init__(self, path):
        self.path = path
        try:
            self.df = pd.read_csv(path)
            print("CSV file opened")
        except:
            try:
                self.df = pd.read_excel(path)
                print("Excel file opened")
            except:
                print("Error: File cannot be opened")

    def pandas_profiling(self):
        self.profile = prof.ProfileReport(self.df)
        self.profile.to_file(output_file="output.html")
        want_preview = 'y'
        if want_preview == 'y':
            self.profile.to_notebook_iframe()
        else:
            pass
        return self.profile

    def preprocess(self):
        self.target_col = 'Target'
        self.df_cat = self.df.select_dtypes(include=['object']).copy()
        self.df_cat = self.df_cat.apply(lambda col: pd.factorize(col, sort=True)[0])
        self.df_cat[self.target_col] = self.df[self.target_col]
        self.inter_df = self.df.drop(self.df_cat.columns, axis=1)
        self.final_df = pd.concat([self.inter_df, self.df_cat], axis=1)
        return self.final_df

    def pycaret(self, dataframe, target_col):
        self.clf1 = setup(data=dataframe, target=target_col, session_id=123, log_experiment=True,
                          experiment_name='hackathon', normalize=True, transformation=True,
                          fix_imbalance=True, preprocess=True,fix_imbalance_method = 'SMOTE')
        self.best_model = compare_models()
        return self.best_model

    def main(self):
        self.preprocess()
        self.pandas_profiling()
        self.pycaret(self.final_df, self.target_col)
        self.real_leaderboard = get_leaderboard()
        self.real_leaderboard.drop(["Model", "MCC", "Kappa"], axis=1, inplace=True)
        return self.real_leaderboard.to_html()

# Define the Gradio input function
def upload_csv_file(csv_file):
    model = PycaretModel(csv_file.name)
    leaderboard = model.main()
    return leaderboard

# Create the Gradio interface
iface = gr.Interface(fn=upload_csv_file, inputs="file", outputs="html", title="Upload CSV , Get best ML model :)")
iface.launch()
