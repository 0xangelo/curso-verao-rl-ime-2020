import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions


def get_optimizer(optimizer, learning_rate, **kwargs):
    optimizers = {
        "adam": tf.keras.optimizers.Adam,
        "nadam": tf.keras.optimizers.Nadam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
    }
    
    return optimizers[optimizer](learning_rate, **kwargs)



def build_discrete_policy(obs_space, action_space, hidden_layers, activation="relu"):
    Input = tf.keras.Input
    Dense = tf.keras.layers.Dense
    DistributionLambda = tfp.layers.DistributionLambda
    Categorical = tfd.Categorical
    
    policy_net_layers = []

    policy_net_layers.append(Input(shape=obs_space.shape, name="State"))

    for i, units in enumerate(hidden_layers):
        policy_net_layers.append(Dense(units=units, activation=activation, name=f"Hidden{i+1}"))
    
    policy_net_layers.append(Dense(units=action_space.n, name="Logits"))
    policy_net_layers.append(DistributionLambda(lambda t: Categorical(logits=t), name="Action_Distribution_Categorical"))
                                     
    return tf.keras.Sequential(policy_net_layers)


def build_continuous_policy(obs_space, action_space, hidden_layers, activation="relu", scale_diag=1e-2):
    Input = tf.keras.Input
    Dense = tf.keras.layers.Dense
    DistributionLambda = tfp.layers.DistributionLambda
    MultivariateNormalDiag = tfd.MultivariateNormalDiag
    
    policy_net_layers = []
    
    policy_net_layers.append(Input(shape=obs_space.shape, name="State"))

    for i, units in enumerate(hidden_layers):
        policy_net_layers.append(Dense(units=units, activation=activation, name=f"Hidden{i+1}"))

    policy_net_layers.append(Dense(units=action_space.shape[0], name="Params"))
    policy_net_layers.append(DistributionLambda(
        lambda t: MultivariateNormalDiag(loc=t, scale_diag=[scale_diag] * action_space.shape[0]),
        name="Action_Distribution_Gaussian"
    ))
                                     
    return tf.keras.Sequential(policy_net_layers)


def build_value_network(obs_space, hidden_layers, activation="relu"):
    Input = tf.keras.Input
    Dense = tf.keras.layers.Dense

    value_net_layers = [ ]

    value_net_layers.append(Input(shape=obs_space.shape, name="State"))

    for i, units in enumerate(hidden_layers):
        value_net_layers.append(Dense(units=units, activation=activation, name=f"Hidden{i+1}"))

    value_net_layers.append(Dense(units=1, name="Value"))

    return tf.keras.Sequential(value_net_layers)
