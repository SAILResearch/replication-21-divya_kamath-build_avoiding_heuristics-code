def batch_bisect(batch_results):
    global batch_total
    
    batch_total += 1
    
    if len(batch_results) == 1:
        return
    
    if 0 in batch_results:
        half_batch = len(batch_results)//2
        batch_bisect(batch_results[:half_batch])
        batch_bisect(batch_results[half_batch:])



def batch_stop_4(batch_results):
    global batch_total
    
    batch_total += 1
    
    if len(batch_results) <= 4:
        if 0 in batch_results:
            batch_total += 4
        return
    
    if 0 in batch_results:
        half_batch = len(batch_results)//2
        batch_stop_4(batch_results[:half_batch])
        batch_stop_4(batch_results[half_batch:])