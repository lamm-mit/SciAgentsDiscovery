import autogen

config_list_4o  = autogen.config_list_from_models(model_list=["gpt-4o"])

config_list_4turbo = autogen.config_list_from_models(model_list=["gpt-4o"])

gpt4o_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.0,
    "config_list": config_list_4o,
    "timeout": 540000,
}


gpt4o_config_graph = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.1,
    "config_list": config_list_4o,
    "timeout": 540000,
    "max_tokens": 2048
}

gpt4turbo_config_graph = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.2,
    "config_list": config_list_4turbo,
    "timeout": 540000,
}

gpt4turbo_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list_4turbo,
    "timeout": 540000,
}