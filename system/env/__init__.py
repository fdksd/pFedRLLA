from gym.envs.registration import register
register(
    id='RFL-v0',
    entry_point='env.env_rfl:RFL',
)