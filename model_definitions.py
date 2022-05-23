from stable_baselines3 import PPO

class Model_Definition:
    def __init__(self,model_constructor,model_kwargs,model_name,model_info="no info"):
        self.model_info = model_info
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.model_constructor = model_constructor

    def get_model(self,env):
        return self.model_constructor(env=env,**self.model_kwargs)

def get_model(env,model_constructor,model_kwargs,model_name,model_info="keine info"):
    return model_constructor(env=env,**model_kwargs),model_name,model_info,PPO.load

def standard_ppo_model(env):
    return PPO(env=env,policy="MlpPolicy"),"standard_ppo","einfach nur ppo mit Standardparametern"


