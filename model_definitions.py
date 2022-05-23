from stable_baselines3 import PPO

def get_model(env,model_constructor,model_kwargs,model_name,model_info="keine info"):
    return model_constructor(env=env,**model_kwargs),model_name,model_info,PPO.load

def standard_ppo_model(env):
    return PPO(env=env,policy="MlpPolicy"),"standard_ppo","einfach nur ppo mit Standardparametern"


