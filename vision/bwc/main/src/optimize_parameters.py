

def optimize_search_direction(video_results, current_start_time, current_end_time, initial_step_size):
    true_timestamps = [float(result['timestamps'][0].split(':')[0]) * 60 + 
                       float(result['timestamps'][0].split(':')[1]) 
                       for result in video_results if result['binary_classification'] == 'TRUE']
    
    if not true_timestamps:
        # No TRUE values found, expand search in both directions with a larger step
        return current_start_time - initial_step_size * 2, current_end_time + initial_step_size * 2

    time_range = current_end_time - current_start_time
    segment_size = time_range / 3
    
    start_segment = current_start_time + segment_size
    end_segment = current_end_time - segment_size

    start_count = sum(1 for t in true_timestamps if t < start_segment)
    middle_count = sum(1 for t in true_timestamps if start_segment <= t < end_segment)
    end_count = sum(1 for t in true_timestamps if t >= end_segment)

    total_true = len(true_timestamps)
    
    # Calculate densities
    start_density = start_count / total_true if total_true > 0 else 0
    middle_density = middle_count / total_true if total_true > 0 else 0
    end_density = end_count / total_true if total_true > 0 else 0

    # Determine step sizes based on densities
    start_step = initial_step_size * (1 + start_density)
    end_step = initial_step_size * (1 + end_density)

    # Adjust start_time and end_time based on densities
    new_start_time = current_start_time
    new_end_time = current_end_time

    if start_density > 0.5:  # More than half of TRUE values in the start segment
        new_start_time -= start_step
    elif end_density > 0.5:  # More than half of TRUE values in the end segment
        new_end_time += end_step
    else:  # TRUE values are more evenly distributed
        new_start_time -= start_step * start_density
        new_end_time += end_step * end_density

    # Ensure we don't overstep in either direction
    new_start_time = max(0, new_start_time)  # Assuming 0 is the minimum possible start time
    new_end_time = min(new_end_time, current_end_time + time_range)  # Limit maximum extension

    return new_start_time, new_end_time