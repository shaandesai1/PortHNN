import models

def get_models(deltat, type=None, hidden_dim=200):
    model_dict = {}
    model_dict['baseline'] = models.base_model(3, hidden_dim, 2, deltat)
    model_dict['HNN'] = models.HNN(2, hidden_dim, 1, deltat)
    model_dict['TDHNN'] = models.TDHNN(3, hidden_dim, 1, deltat)
    model_dict['TDHNN4'] = models.TDHNN4(2, hidden_dim, 1, deltat)

    if type is None:
        return model_dict

    elif type in model_dict.keys():
        return model_dict[type]

    else:
        raise ValueError('type not understood')
