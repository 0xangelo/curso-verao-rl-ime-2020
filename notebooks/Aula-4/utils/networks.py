import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions


def build_actor_critic_network(obs_space, action_space, config):
    Input = tf.keras.Input
    Dense = tf.keras.layers.Dense
    DistributionLambda = tfp.layers.DistributionLambda
    Categorical = tfd.Categorical

    hidden_layers = config["hidden_layers"]
    activation = config["activation"]

    inputs = Input(shape=obs_space.shape)

    h = inputs
    for i, units in enumerate(hidden_layers):
        h = Dense(units, activation, name=f"Hidden{i+1}")(h)

    n_actions = action_space.n

    logits = Dense(n_actions, name="logits")(h)
    action_dist = DistributionLambda(lambda t: Categorical(logits=t))(logits)
    value_fn = Dense(units=1, name="ValueFn")(h)

    return tf.keras.Model(inputs=inputs, outputs=(action_dist, value_fn))
