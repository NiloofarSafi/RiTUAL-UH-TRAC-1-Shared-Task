import pandas as pd


def show_mistakes(fname, X, y_act, y_pred):
    mistake_ind = [y_act != y_pred]
    mistakes = zip(X[mistake_ind], y_act[mistake_ind], y_pred[mistake_ind])
    # Classes
    # are: cag, nag, oag

    data = []
    for mistake, act, pred in mistakes:
        if act == 0:
            actl = 'cag'
        elif act == 1:
            actl = 'nag'
        else:
            actl = 'oag'

        if pred==0:
            predl = 'cag'
        elif pred ==1:
            predl = 'nag'
        else:
            predl = 'oag'
        row = {'id': mistake.id, 'content': mistake.content, 'actual': actl,
               'predicted': predl}
        data.append(row)

    df = pd.DataFrame(data,columns=['id','type','content','actual','predicted'])
    df.to_csv(fname, encoding='utf=8',index=False)
