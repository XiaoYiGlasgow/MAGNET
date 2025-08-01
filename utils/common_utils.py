# 简化版 common_utils.py
def get_scenario_map():
    """返回场景映射"""
    return {
        'following_lane_with_lead': [15.0, -3.0],
        'changing_lane': [15.0, -3.0],
        'starting_left_turn': [15.0, -3.0],
        'starting_right_turn': [15.0, -3.0],
        'stopping_with_lead': [15.0, -3.0],
    }

def get_filter_parameters(num_scenarios_per_type=1, limit_total_scenarios=3, shuffle=False):
    """返回过滤参数"""
    scenario_types = [
        'following_lane_with_lead',
        'changing_lane', 
        'starting_left_turn',
        'starting_right_turn',
        'stopping_with_lead'
    ]
    
    return (
        scenario_types[:3],  # 只取前3种类型
        None,  # scenario_tokens
        None,  # log_names  
        None,  # map_names
        num_scenarios_per_type,
        limit_total_scenarios,
        None,  # timestamp_threshold_s
        None,  # ego_displacement_minimum_m
        False, # expand_scenarios
        True,  # remove_invalid_goals
        shuffle,
        None,  # ego_start_speed_threshold
        None,  # ego_stop_speed_threshold
        None   # speed_noise_tolerance
    )
