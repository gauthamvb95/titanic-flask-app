import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import pickle


def extract_cabin_letters(df_all):
    print(type(df_all))
    df_all = df_all.fillna({"Cabin": "M"})
    df_all['cabin_letter'] = df_all['Cabin'].apply(lambda x: x[0])
    return df_all


def avg_age(df_all):
    col_age = {}
    for col in ['is_Mr', 'is_Mrs', 'is_Miss', 'is_Master', 'is_Dr', 'is_Rev']:
        col_age.update({col: df_all[df_all[col] == 1]['Age'].mean()})
    return col_age


def addn_cols(df_all):
    df_all = extract_cabin_letters(df_all)
    df_all['is_cabin_BDE'] = df_all['cabin_letter'].apply(lambda x: 1 if ((x == 'B') | (x == 'D') | (x == 'E')) else 0)
    df_all['is_parent_or_child'] = df_all['Parch'].apply(lambda x: 1 if x > 0 else 0)
    df_all['is_sibling_or_spouse'] = df_all['SibSp'].apply(lambda x: 1 if x > 0 else 0)
    df_all['Ticket_Lett'] = df_all['Ticket'].apply(lambda x: str(x)[0])
    df_all['is_tick_let_1'] = df_all['Ticket_Lett'].apply(lambda x: 1 if x == '1' else 0)
    df_all['is_tick_let_2'] = df_all['Ticket_Lett'].apply(lambda x: 1 if x == '2' else 0)
    df_all['is_tick_let_3'] = df_all['Ticket_Lett'].apply(lambda x: 1 if x == '3' else 0)
    df_all['is_tick_let_P'] = df_all['Ticket_Lett'].apply(lambda x: 1 if x == 'P' else 0)
    df_all['is_male'] = df_all['Sex'].apply(lambda x: 1 if x == "male" else 0)
    df_all['is_Mr'] = df_all['Name'].str.contains('Mr\.').apply(lambda x: 1 if x == True else 0)
    df_all['is_Mrs'] = df_all['Name'].str.contains('Mrs\.').apply(lambda x: 1 if x == True else 0)
    df_all['is_Miss'] = df_all['Name'].str.contains('Miss\.').apply(lambda x: 1 if x == True else 0) | df_all[
        'Name'].str.contains('Ms\.').apply(lambda x: 1 if x == True else 0)
    df_all['is_Master'] = df_all['Name'].str.contains('Master\.').apply(lambda x: 1 if x == True else 0)
    df_all['is_Dr'] = df_all['Name'].str.contains('Dr\.').apply(lambda x: 1 if x == True else 0)
    df_all['is_Rev'] = df_all['Name'].str.contains('Rev\.').apply(lambda x: 1 if x == True else 0)
    col_age = avg_age(df_all)
    for col in ['is_Mr', 'is_Mrs', 'is_Miss', 'is_Master', 'is_Dr', 'is_Rev']:
        df_all['Age'] = np.where((df_all.Age.isnull()) & (df_all[col] == 1), col_age[col], df_all.Age)
    df_all['Embarked'] = np.where((df_all.Embarked.isnull()), 'S', df_all.Embarked)
    df_all['is_Pclass_1'] = df_all['Pclass'].apply(lambda x: 1 if x == 1 else 0)
    df_all['is_Pclass_2'] = df_all['Pclass'].apply(lambda x: 1 if x == 2 else 0)
    df_all['is_embark_C'] = df_all['Embarked'].apply(lambda x: 1 if x == 'C' else 0)
    df_all['is_embark_S'] = df_all['Embarked'].apply(lambda x: 1 if x == 'S' else 0)
    df_all['is_Age_less_than_9'] = df_all['Age'].apply(lambda x: 1 if x < 9 else 0)
    return df_all


def return_train_test(df_all):
    final_train = df_all[df_all['Survived'].notnull()]
    final_test = df_all[df_all['Survived'].isnull()]
    return final_train, final_test


def return_rf_model(final_train, selected_features):
    rf = RandomForestClassifier(criterion='gini',
                                n_estimators=700,
                                min_samples_split=10,
                                min_samples_leaf=1,
                                max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

    X = final_train[selected_features]
    y = final_train['Survived']
    return rf.fit(X, y)


def main():
    file_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    train = pd.read_csv(os.path.join(file_path, "data\\train.csv"))
    test = pd.read_csv(os.path.join(file_path, "data\\test.csv"))
    df_all = pd.concat([train, test], sort=True).reset_index(drop=True)
    df_all = addn_cols(df_all)
    selected_features = ['is_Mr', 'is_Mrs', 'is_Miss', 'is_Master', 'is_Pclass_1', 'is_Pclass_2', 'is_embark_S',
                         'is_embark_C', 'is_parent_or_child']
    final_train, final_test = return_train_test(df_all)

    model = return_rf_model(final_train, selected_features)
    my_path = Path(os.path.join(file_path, "models\\titanic_rf.pkl"))
    with my_path.open('wb') as fp:
        pickle.dump(model, fp)


if __name__ == '__main__':
    main()
