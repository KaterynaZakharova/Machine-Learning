import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def blight_model():
    train = pd.read_csv('datasets/train/train.csv', encoding="ISO-8859-1")
    test = pd.read_csv('datasets/test/test.csv', encoding="ISO-8859-1")
    addresses = pd.read_csv('datasets/addresses.csv', encoding="ISO-8859-1")

    train = pd.merge(train, addresses, left_on='ticket_id', right_on='ticket_id')
    test = pd.merge(test, addresses, left_on='ticket_id', right_on='ticket_id')

    train = train.dropna(subset=['compliance'])
    train['compliance'] = train['compliance'].astype(int)
    column_dict = {'compliance': 'category', 'country': 'category', 'non_us_str_code': 'category', 'state': 'category',
                   'zip_code': 'category'}

    for df in [test, train]:
        for column, column_type in column_dict.items():
            if column in df:
                if column_type == 'category':
                    df[column] = df[column].replace(np.nan, "NAN", regex=True).astype('category')
                elif column_type == 'int':
                    df[column] = df[column].replace(np.nan, 0, regex=True).astype(int)

    train = train.drop(['address', 'admin_fee', 'agency_name', 'city', 'clean_up_cost', 'disposition', 'fine_amount',
                        'grafitti_status', 'hearing_date', 'inspector_name', 'late_fee', 'mailing_address_str_name',
                        'mailing_address_str_number', 'state_fee', 'ticket_issued_date', 'violation_code',
                        'violation_description', 'violation_street_name', 'violation_street_number',
                        'violation_zip_code',
                        'violator_name', 'balance_due', 'payment_amount', 'payment_date', 'payment_status'],
                       axis=1).set_index('ticket_id')

    test = test.drop(['address', 'admin_fee', 'agency_name', 'city', 'clean_up_cost', 'disposition', 'fine_amount',
                      'grafitti_status', 'hearing_date', 'inspector_name', 'late_fee', 'mailing_address_str_name',
                      'mailing_address_str_number', 'state_fee', 'ticket_issued_date', 'violation_code',
                      'violation_description', 'violation_street_name', 'violation_street_number', 'violation_zip_code',
                      'violator_name'], axis=1).set_index('ticket_id')

    y_train = train['compliance']

    X_train_to_drop = ['compliance', 'compliance_detail', 'collection_status']
    train = train.drop(X_train_to_drop, axis=1)

    category_columns = train.select_dtypes(['category']).columns

    for df in [test, train]:
        df[category_columns] = df[category_columns].apply(lambda x: x.cat.codes)

    X_train = train.copy()

    grid_values = {'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 4, 5]}

    clf = GradientBoostingClassifier(random_state=0)
    grid_clf = GridSearchCV(clf, param_grid=grid_values, scoring='roc_auc')
    grid_clf.fit(X_train, y_train)

    pred = grid_clf.predict_proba(test)[:, 1]

    res = pd.Series(name='compliance', data=pred, index=test.index, dtype='float32')

    return res


print(blight_model())
