import numpy as np
import pandas as pd
from typing import List, Dict


def generate_type_a_sessions(n: int, rng: np.random.Generator) -> List[Dict]:
    """
    Generate synthetic time-series data for Type A users.

    Args:
        n (int): Number of user sessions to generate.
        rng (np.random.Generator): Numpy generator object.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a user session.
                    Each session contains a list of time steps with the specified variables.
    """
    
    sessions = []

    possible_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    weights = [1, 1, 1, 2, 1, 1, 1, 3, 2, 2, 2, 5, 4, 4, 4, 7]

    for _ in range(n):
        session_length = rng.choice(possible_lengths, p=np.array(weights) / np.sum(weights))

        session_data = {
            "TimeFocused": [],
            "TimeDiverted": [],
            "TimeDistracted": [],
            "NumDiverted": [],
            "NumDistracted": [],
            "NumNotifs": [],
            "AvgFriendsPerMin": [],
            "FirstDistraction": [],
            "FirstDiversion": []
        }

        avg_friends = rng.choice([0, 1, 2, 3, 4], p=[0.8, 0.1, 0.05, 0.025, 0.025])

        avg_friends_per_min = []
        for step in range(session_length):
            friends_change = False

            if avg_friends > 0:
                if rng.random() < 0.1:
                    avg_friends -= 1
                    friends_change = True
            else:
                if rng.random() < 0.05:
                    avg_friends = 1
                    friends_change = True

            # add noise to AvgFriendsPerMin to simulate friends leaving in middle of time step
            if friends_change:
                noise = rng.uniform(-0.1, 0.1)
                avg_friends_step = max(0, min(1, avg_friends + noise))
            else:
                avg_friends_step = avg_friends
                
            avg_friends_per_min.append(avg_friends_step)

        total_distracted = rng.poisson(lam=0.5 * (session_length / 4))  
        distracted_indices = np.linspace(0, session_length - 1, total_distracted, dtype=int)

        total_diverted = rng.poisson(lam=0.5 * (session_length / 4))
        diverted_indices = np.linspace(0, session_length - 1, total_diverted, dtype=int)

        for step in range(session_length):
            if step in distracted_indices:
                time_distracted = rng.integers(1, 300)
                num_distracted = 1
                first_distraction = rng.choice(["social", "game", "video"], p=[0.7, 0.2, 0.1])
            else:
                time_distracted = 0
                num_distracted = 0
                first_distraction = "none"

            if step in diverted_indices:
                time_diverted = rng.integers(1, 30)
                num_diverted = 1

                if avg_friends_per_min[step] > 0:
                    first_diversion = rng.choice(["timer", "progress", "friends", "customizations"], p=[0.5, 0.3, 0.15, 0.05])
                else:
                    first_diversion = rng.choice(["timer", "progress", "customizations"], p=[0.55, 0.4, 0.05])
            else:
                time_diverted = 0
                num_diverted = 0
                first_diversion = "none"

            time_focused = 900 - time_diverted - time_distracted

            num_notifs = rng.integers(0, 16)

            session_data["TimeFocused"].append(time_focused)
            session_data["TimeDiverted"].append(time_diverted)
            session_data["TimeDistracted"].append(time_distracted)
            session_data["NumDiverted"].append(num_diverted)
            session_data["NumDistracted"].append(num_distracted)
            session_data["NumNotifs"].append(num_notifs)
            session_data["AvgFriendsPerMin"].append(avg_friends_per_min[step])
            session_data["FirstDistraction"].append(first_distraction)
            session_data["FirstDiversion"].append(first_diversion)

        sessions.append(session_data)

    return sessions


def generate_type_b_sessions(n: int, rng: np.random.Generator) -> List[Dict]:
    """
    Generate synthetic time-series data for Type B users.

    Args:
        n (int): Number of user sessions to generate.
        rng (np.random.Generator): Numpy generator object.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a user session.
                    Each session contains a list of time steps with the specified variables.
    """

    sessions = []
    possible_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    for _ in range(n):
        avg_friends = rng.choice([0, 1, 2, 3, 4], p=[0.05, 0.2, 0.3, 0.3, 0.15])
        base_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # scale weights based on avg_friends and session length
        length_factor = 0.05
        scaled_weights = base_weights * (1 + (avg_friends * length_factor * np.array(possible_lengths)))
        scaled_weights = scaled_weights / np.sum(scaled_weights)
        # sessions with (more) friends tend to be longer
        session_length = rng.choice(possible_lengths, p=scaled_weights)

        session_data = {
            "TimeFocused": [],
            "TimeDiverted": [],
            "TimeDistracted": [],
            "NumDiverted": [],
            "NumDistracted": [],
            "NumNotifs": [],
            "AvgFriendsPerMin": [],
            "FirstDistraction": [],
            "FirstDiversion": []
        }

        avg_friends_per_min = []
        for step in range(session_length):
            friends_change = False

            if avg_friends == 2 or avg_friends == 3:
                if rng.random() < 0.05:
                    avg_friends -= 1
                    friends_change = True
                if rng.random() < 0.05:
                    avg_friends += 1
                    friends_change = True
            
            elif avg_friends == 0:
                if rng.random() < 0.6:
                    avg_friends = 1
                    friends_change = True
            
            elif avg_friends == 1:
                if rng.random() < 0.01:
                    avg_friends -= 1
                    friends_change = True
                if rng.random() < 0.3:
                    avg_friends += 1
                    friends_change = True
            
            else:
                if rng.random() < 0.05:
                    avg_friends -= 1
                    friends_change = True

            # add noise to AvgFriendsPerMin to simulate friends leaving in middle of time step
            if friends_change:
                noise = rng.uniform(-0.1, 0.1)
                avg_friends_step = max(0, min(4, avg_friends + noise))
            else:
                avg_friends_step = avg_friends
                
            avg_friends_per_min.append(avg_friends_step)


        if avg_friends == 0: # if alone, more distractions/diversions
            total_distracted = rng.poisson(lam=0.8 * (session_length / 4))
            total_diverted = rng.poisson(lam=0.8 * (session_length / 4))
        else: # if with friends, fewer distractions/divertations as avg_friends increases
            total_distracted = rng.poisson(lam=0.8 * (session_length / 4) * (1 - avg_friends * 0.2))
            total_diverted = rng.poisson(lam=0.8 * (session_length / 4) * (1 - avg_friends * 0.2))

        total_distracted = max(0, total_distracted)
        total_diverted = max(0, total_diverted)

        distracted_indices = rng.choice(
            session_length, size=min(total_distracted, session_length), replace=False
        )
        diverted_indices = rng.choice(
            session_length, size=min(total_diverted, session_length), replace=False
        )

        for step in range(session_length):
            if step in distracted_indices:
                time_distracted = rng.integers(3, 300)
                num_distracted = max(1, round(time_distracted / 300))  # approximately 1 per 300 seconds
                first_distraction = rng.choice(["social", "game", "video"], p=[0.8, 0.1, 0.1])
            else:
                time_distracted = 0
                num_distracted = 0
                first_distraction = "none"

            if step in diverted_indices:
                time_diverted = rng.integers(1, 300)
                num_diverted = max(1, round(time_diverted / 300))  # approximately 1 per 300 seconds

                if avg_friends_per_min[step] > 0:
                    first_diversion = rng.choice(["timer", "progress", "friends", "customizations"], p=[0.25, 0.25, 0.4, 0.1])
                else:
                    first_diversion = rng.choice(["timer", "progress", "customizations"], p=[0.4, 0.5, 0.1])
            else:
                time_diverted = 0
                num_diverted = 0
                first_diversion = "none"

            time_focused = 900 - time_diverted - time_distracted

            num_notifs = rng.integers(0, 31)

            session_data["TimeFocused"].append(time_focused)
            session_data["TimeDiverted"].append(time_diverted)
            session_data["TimeDistracted"].append(time_distracted)
            session_data["NumDiverted"].append(num_diverted)
            session_data["NumDistracted"].append(num_distracted)
            session_data["NumNotifs"].append(num_notifs)
            session_data["AvgFriendsPerMin"].append(avg_friends_per_min[step])
            session_data["FirstDistraction"].append(first_distraction)
            session_data["FirstDiversion"].append(first_diversion)

        sessions.append(session_data)

    return sessions


def generate_type_c_sessions(n: int, rng: np.random.Generator) -> List[Dict]:
    """
    Generate synthetic time-series data for Type C users.

    Args:
        n (int): Number of user sessions to generate.
        rng (np.random.Generator): Numpy generator object.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a user session.
                    Each session contains a list of time steps with the specified variables.
    """

    sessions = []
    possible_lengths = list(range(1, 17))

    for _ in range(n):
        session_length = rng.choice(possible_lengths)
        total_time = session_length * 900
        min_not_focused = total_time // 3
        total_not_focused = rng.integers(min_not_focused, total_time + 1)

        diverted_indices, distracted_indices = [], []
        for step in range(session_length):
            if step > 0 and step - 1 in diverted_indices:
                if rng.random() < 0.6:
                    distracted_indices.append(step)
            elif step > 0 and step - 1 in distracted_indices:
                continue
            else:
                if rng.random() < 0.5:
                    if rng.random() < 0.5:
                        diverted_indices.append(step)
                    else:
                        distracted_indices.append(step)

        time_diverted_total = rng.integers(0, total_not_focused + 1)
        time_distracted_total = total_not_focused - time_diverted_total

        step_times = distribute_time_fixed(session_length, diverted_indices, distracted_indices, time_diverted_total, time_distracted_total, rng)
        session_data = {key: [] for key in ["TimeFocused", "TimeDiverted", "TimeDistracted", "NumDiverted", "NumDistracted", "NumNotifs", "AvgFriendsPerMin", "FirstDistraction", "FirstDiversion"]}

        avg_friends = rng.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.35, 0.35])

        for step in range(session_length):
            time_focused, time_diverted, time_distracted = step_times[step].values()
            num_diverted = rng.integers(1, 4) if time_diverted > 0 else 0
            num_distracted = rng.integers(1, 4) if time_distracted > 0 else 0
            num_notifs = rng.integers(5, 31)

            first_diversion = rng.choice(["timer", "progress", "friends", "customizations"], p=[0.1, 0.1, 0.4, 0.4]) if time_diverted > 0 else "none"
            first_distraction = rng.choice(["social", "game", "video"], p=[0.8, 0.1, 0.1]) if time_distracted > 0 else "none"

            avg_friends_per_min = 0
            friends_change = False

            if avg_friends in [2, 3]:
                if rng.random() < 0.05:
                    avg_friends -= 1
                    friends_change = True
                if rng.random() < 0.05:
                    avg_friends += 1
                    friends_change = True
            elif avg_friends == 1:
                if rng.random() < 0.01:
                    avg_friends -= 1
                    friends_change = True
                if rng.random() < 0.3:
                    avg_friends += 1
                    friends_change = True
            else:
                if rng.random() < 0.05:
                    avg_friends -= 1
                    friends_change = True

            if friends_change:
                noise = rng.uniform(-0.2, 0.2)
                avg_friends_per_min = max(1.0, min(4.0, avg_friends + noise))
            else:
                avg_friends_per_min = avg_friends

            session_data["TimeFocused"].append(time_focused)
            session_data["TimeDiverted"].append(time_diverted)
            session_data["TimeDistracted"].append(time_distracted)
            session_data["NumDiverted"].append(num_diverted)
            session_data["NumDistracted"].append(num_distracted)
            session_data["NumNotifs"].append(num_notifs)
            session_data["AvgFriendsPerMin"].append(avg_friends_per_min)
            session_data["FirstDistraction"].append(first_distraction)
            session_data["FirstDiversion"].append(first_diversion)

        sessions.append(session_data)
    
    return sessions


def distribute_time_fixed(session_length: int, diverted_indices: List[int], distracted_indices: List[int], time_diverted_total: int, time_distracted_total: int, rng: np.random.Generator) -> Dict[int, Dict[str, int]]:
    """
    Distribute total time spent in the 3 activities across time steps in a session for Type C users.
    
    Args:
        session_length (int): Number of time steps in session.
        diverted_indices (List[int]): Indices of time steps where a Diversion occurred.
        distracted_indices (List[int]): Indices of time steps where a Distraction occurred.
        time_diverted_total (int): Total TimeDiverted of the session.
        time_distracted_total (int): Total TimeDistracted the session.
        rng (np.random.Generator): Numpy generator object. If None, random seed is used.
    
    Returns:
        step_times (Dict[int, Dict[str, int]]): Dictionary mapping each time step to a dictionary of time spent in each activity.
    """
    
    step_times = {step: {"TimeFocused": 900, "TimeDiverted": 0, "TimeDistracted": 0} for step in range(session_length)}
    
    def distribute_amount(total: int, indices: List[int], rng) -> Dict[int, int]:
        allocation = {idx: 0 for idx in indices}
        remaining_total = total

        while remaining_total > 0 and indices:
            idx = rng.choice(indices)
            max_add = min(remaining_total, 900 - step_times[idx]["TimeDiverted"] - step_times[idx]["TimeDistracted"])
            if max_add > 0:
                add_amount = rng.integers(1, max_add + 1) if max_add > 1 else 1
                allocation[idx] += add_amount
                step_times[idx]["TimeDiverted" if idx in diverted_indices else "TimeDistracted"] += add_amount
                remaining_total -= add_amount
            else:
                indices.remove(idx)
        
        return allocation
    
    distribute_amount(time_diverted_total, diverted_indices, rng)
    distribute_amount(time_distracted_total, distracted_indices, rng)
    
    for step in range(session_length):
        step_times[step]["TimeFocused"] = 900 - step_times[step]["TimeDiverted"] - step_times[step]["TimeDistracted"]
        assert step_times[step]["TimeFocused"] >= 0, "TimeFocused cannot be negative"
    
    return step_times


def generate_type_d_sessions(n: int, rng: np.random.Generator) -> List[Dict]:
    """
    Generate synthetic time-series data for Type D users.

    Args:
        n (int): Number of user sessions to generate.
        rng (np.random.Generator): Numpy generator object.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a user session.
                    Each session contains a list of time steps with the specified variables.
    """
    
    sessions = []

    possible_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    weights = [4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2]

    for _ in range(n):
        session_length = rng.choice(possible_lengths, p=np.array(weights) / np.sum(weights))

        session_data = {
            "TimeFocused": [],
            "TimeDiverted": [],
            "TimeDistracted": [],
            "NumDiverted": [],
            "NumDistracted": [],
            "NumNotifs": [],
            "AvgFriendsPerMin": [],
            "FirstDistraction": [],
            "FirstDiversion": []
        }

        avg_friends = rng.choice([0, 1, 2, 3, 4])

        avg_friends_per_min = []
        for step in range(session_length):
            if rng.random() < 0.5:
                if avg_friends == 0:
                    avg_friends += rng.choice([1, 2, 3, 4])
                elif avg_friends == 1:
                    avg_friends += rng.choice([-1, 1, 2, 3])
                elif avg_friends == 2:
                    avg_friends += rng.choice([-2, -1, 1, 2])
                elif avg_friends == 3:
                    avg_friends += rng.choice([-3, -2, -1, 1])
                elif avg_friends == 4:
                    avg_friends += rng.choice([-4, -3, -2, -1])

                noise = rng.uniform(-0.1, 0.1)
                avg_friends_step = max(0, min(4, avg_friends + noise))
            else:
                avg_friends_step = avg_friends

            avg_friends_per_min.append(avg_friends_step)

        isDistracted = False
        for step in range(session_length):
            # case where previous step had 2+ notifs, gauranteeing this step has TimeDistracted >= 300
            if isDistracted:
                time_distracted = rng.integers(300, 901)
                num_distracted = rng.integers(1, 6) if time_distracted < 900 else 1
                
                time_focused = rng.integers(0, 901 - time_distracted)
                time_diverted = 900 - time_focused - time_distracted
                num_diverted = rng.integers(1, 6) if time_diverted > 0 else 0

                isDistracted = False
            # regular case
            else:
                time_focused = rng.integers(2, 451)
                time_diverted = rng.integers(2, 451)
                time_distracted = 900 - time_focused - time_diverted
                num_distracted = rng.integers(1, 6) if time_distracted > 0 else 0
                num_diverted = rng.integers(1, 6)
                
            num_notifs = rng.integers(0, 11)
            if num_notifs >- 2:
                isDistracted = True

            first_distraction = rng.choice(["social", "game", "video"], p=[0.34, 0.33, 0.33])

            if avg_friends_per_min[step] > 0:
                first_diversion = rng.choice(["timer", "progress", "friends", "customizations"], p=[0.25, 0.25, 0.25, 0.25])
            else:
                first_diversion = rng.choice(["timer", "progress", "customizations"], p=[0.33, 0.34, 0.33])

            session_data["TimeFocused"].append(time_focused)
            session_data["TimeDiverted"].append(time_diverted)
            session_data["TimeDistracted"].append(time_distracted)
            session_data["NumDiverted"].append(num_diverted)
            session_data["NumDistracted"].append(num_distracted)
            session_data["NumNotifs"].append(num_notifs)
            session_data["AvgFriendsPerMin"].append(avg_friends_per_min[step])
            session_data["FirstDistraction"].append(first_distraction)
            session_data["FirstDiversion"].append(first_diversion)

        sessions.append(session_data)

    return sessions


def transform_time_series_df(df: pd.DataFrame, user_type: str) -> pd.DataFrame:
    """
    Transform a DataFrame of time-series data into a new DataFrame where each row represents a single time step.

    Args:
        df (pd.DataFrame): The input DataFrame, where each row contains arrays representing time-series data.
        user_type (str): User type to assign to all rows in the output DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame where each row represents a single time step for a unique session.
    """
    new_rows = []

    for session_id, session_data in df.iterrows():
        num_steps = len(session_data["TimeFocused"])
        
        for time_step in range(num_steps):
            new_row = {
                "Session": session_id,
                "TimeStep": time_step, 
                "TimeFocused": session_data["TimeFocused"][time_step],
                "TimeDiverted": session_data["TimeDiverted"][time_step],
                "TimeDistracted": session_data["TimeDistracted"][time_step],
                "NumDiverted": session_data["NumDiverted"][time_step],
                "NumDistracted": session_data["NumDistracted"][time_step],
                "NumNotifs": session_data["NumNotifs"][time_step],
                "AvgFriendsPerMin": session_data["AvgFriendsPerMin"][time_step],
                "FirstDistraction": session_data["FirstDistraction"][time_step],
                "FirstDiversion": session_data["FirstDiversion"][time_step],
                "Type": user_type
            }
            
            new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)

    new_df = new_df[["Session", "TimeStep", "TimeFocused", "TimeDiverted", "TimeDistracted", 
                     "NumDiverted", "NumDistracted", "NumNotifs", "AvgFriendsPerMin", 
                     "FirstDistraction", "FirstDiversion", "Type"]]

    return new_df