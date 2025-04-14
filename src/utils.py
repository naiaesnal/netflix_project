import pandas as pd
from itertools import product
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

# Returns a data frame with errors for uknn and iknn models
def get_uknn_iknn_errors(model, df):
    real_pairs = zip(df['userid'], df['itemid'])
    predictions = []
    
    for user_id, item_id, real_rating in zip(df['userid'], df['itemid'], df['rating']):
        try:
            score = model.score(user_id, item_id)
            if score is not None:
                predictions.append({
                    'userid': user_id,
                    'itemid': item_id,
                    'real_rating': real_rating,
                    'predicted_rating': score,
                    'error': real_rating - score
                })
        except:
            continue
            
    return pd.DataFrame(predictions)

# Predicts ratings for missing user-item pairs for uknn and iknn models
def get_uknn_iknn_scores(model, user_ids, item_ids, existing_pairs):
    scores = {}
    for user_id, item_id in product(user_ids, item_ids):
        if (user_id, item_id) not in existing_pairs:
            try:
                score = model.score(user_id, item_id)
                if score is not None:
                    scores[(user_id, item_id)] = score
            except:
                continue  # Ignores errors (not seen users/items)
    return scores

# Predicts ratings for missing user-item pairs for svd model
def get_svd_predictions(model, missing_pairs):
    for user_id, item_id in missing_pairs:
        estimated_rating = model.predict(user_id, item_id).est
        yield {'userid': user_id, 'itemid': item_id, 'estimated_rating': estimated_rating}

# Plots precision-recall curve for als model
def plot_precision_recall_curve(model, df, user_map, item_map):
    y_true = []
    y_scores = []

    real_interactions = set(zip(df['userid'], df['itemid']))

    for user_id in user_map:
        for item_id in item_map:
            # Verify interaction
            real_interaction = 1 if (user_id, item_id) in real_interactions else 0
            y_true.append(real_interaction)

            # Obtain latent values
            user_vector = model.user_factors[user_map[user_id]]
            item_vector = model.item_factors[item_map[item_id]]
            predicted_score = np.dot(user_vector, item_vector)
            y_scores.append(predicted_score)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.show()