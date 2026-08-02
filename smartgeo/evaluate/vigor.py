import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict_vigor


def validate_top(grd_descriptor, sat_descriptor, query_labels, step_size):

    top1_percent = sat_descriptor.shape[0] // 100
    results = np.zeros(top1_percent)

    accuracy_top1p = 0.0
    accuracy_top1 = 0.0
    accuracy_top5 = 0.0
    accuracy_top10 = 0.0
    accuracy_hit = 0.0

    data_amount = 0.0
    # dist_array = 2 - 2 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))
    # dist_array = 2 - 2 * grd_descriptor @ sat_descriptor.T
    
    steps = len(grd_descriptor) // step_size + 1
    for split in range(steps):
        grd_descriptor_i = grd_descriptor[(split*step_size):((split+1)*step_size), :]
        query_labels_i = query_labels[(split*step_size):((split+1)*step_size)]
        similarity = (grd_descriptor_i @ sat_descriptor.T).cpu().numpy()
        for i in range(grd_descriptor_i.shape[0]):
            gt_dist = similarity[i, query_labels_i[i][0].cpu()]
            ranking = np.sum(similarity[i, :] > gt_dist)
            dist_temp = np.ones(similarity[i, :].shape[0])
            dist_temp[query_labels_i[i][1:].cpu()] = 0
            prediction_hit = np.sum((similarity[i, :] > gt_dist) * dist_temp)
            if ranking < top1_percent:
                accuracy_top1p += 1.0
            if ranking < 1:
                accuracy_top1 += 1.0
            if ranking < 5:
                accuracy_top5 += 1.0
            if ranking < 10:
                accuracy_top10 += 1.0
            if prediction_hit < 1:
                accuracy_hit += 1.0
            data_amount += 1.0

    accuracy_top1p /= data_amount /100
    accuracy_top1 /= data_amount /100
    accuracy_top5 /= data_amount /100
    accuracy_top10 /= data_amount /100
    accuracy_hit /= data_amount /100

    print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1p:%.2f' % (accuracy_top1, accuracy_top5, accuracy_top10, accuracy_top1p))
    print('Hit rate:%.2f'% (accuracy_hit)) 
    return [accuracy_top1, accuracy_top5, accuracy_top10, accuracy_top1p, accuracy_hit]

def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=3000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict_vigor(config, model, reference_dataloader, 'reference') 
    query_features, query_labels = predict_vigor(config, model, query_dataloader, 'query')
    
    
    print("Compute Scores:")
    result, hit =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 

    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, query_features, query_labels
        gc.collect()
        
    return result, hit


def calc_sim(config,
                        model,
                        reference_dataloader,
                        query_dataloader, 
                        ranks=[1, 5, 10],
                        step_size=3000,
                        cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict_vigor(config, model, reference_dataloader, 'reference') 
    query_features, query_labels = predict_vigor(config, model, query_dataloader, 'query')
    
    # print("Compute Scores Train:")
    # r1 =  calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    # acc = validate_top(query_features, reference_features, query_labels, step_size)
    print("Similarity Sampling:")
    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return near_dict



def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=8000, ranks=[1, 5, 10]):
    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    # Move labels to CPU
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    # Create a dictionary to map reference labels to indices
    ref2index = {idx: i for i, idx in enumerate(reference_labels_np)}
    
    # Initialize results
    topk.append(R // 100)
    results = np.zeros(len(topk))
    hit_rate = 0.0
    
    # Process in chunks to reduce memory usage
    for i in tqdm(range(Q)):
        # Compute similarity for the current query
        start = i
        end = i + 1
        sim_tmp = query_features[start:end] @ reference_features.T  # Shape: (1, R)
        
        # Move similarity to CPU
        sim_tmp = sim_tmp.cpu()
        
        # Get the similarity value of the ground truth reference
        gt_sim = sim_tmp[0, ref2index[query_labels_np[i][0]]]
        
        # Number of references with higher similarity than the ground truth
        higher_sim = sim_tmp > gt_sim
        
        # Calculate ranking
        ranking = higher_sim.sum().item()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.0
        
        # Mask for semi-positive references
        mask = torch.ones(R, dtype=torch.bool)
        for near_pos in query_labels_np[i][1:]:
            mask[ref2index[near_pos]] = False
        
        # Calculate hit rate
        hit = (higher_sim * mask).sum().item()
        if hit < 1:
            hit_rate += 1.0
    
    # Normalize results
    results = results / Q * 100.0
    hit_rate = hit_rate / Q * 100.0
    
    # Print results
    string = [f'Recall@{k}: {results[i]:.4f}' for i, k in enumerate(topk[:-1])]
    string.append(f'Recall@top1: {results[-1]:.4f}')
    string.append(f'Hit_Rate: {hit_rate:.4f}')
    print(' - '.join(string))
    
    return results, hit_rate

def calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=8000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    query_labels_np = query_labels[:,0].cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
        
    results = results/ Q * 100.

    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))           
        
    print(' - '.join(string)) 

    return results[0]
   

def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):

    query_labels = query_labels[:,0]
    
    Q = len(query_features)
    
    steps = Q // step_size + 1

    # Initialize a tensor to store topk scores and indices
    topk_scores = torch.zeros((Q, neighbour_range + 2), device=query_features.device)
    topk_ids = torch.zeros((Q, neighbour_range + 2), dtype=torch.long, device=query_features.device)

    # Process in chunks to reduce memory usage
    for i in range(steps):
        start = step_size * i
        end = min(start + step_size, Q)

        # Compute similarity for the current chunk
        sim_tmp = query_features[start:end] @ reference_features.T

        # Find topk for the current chunk
        chunk_topk_scores, chunk_topk_ids = torch.topk(sim_tmp, k=neighbour_range + 2, dim=1)

        # Store the results
        topk_scores[start:end] = chunk_topk_scores
        topk_ids[start:end] = chunk_topk_ids

        # Free memory
        del sim_tmp, chunk_topk_scores, chunk_topk_ids

    # similarity = []
    
    # for i in range(steps):
        
    #     start = step_size * i
        
    #     end = start + step_size
          
    #     sim_tmp = query_features[start:end] @ reference_features.T
        
    #     similarity.append(sim_tmp.cpu())
     
    # # matrix Q x R
    # similarity = torch.cat(similarity, dim=0)

    # # there might be more ground views for same sat view
    # topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+2, dim=1)

    topk_references = []
    
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
    topk_references = torch.stack(topk_references, dim=0)

     
    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)
    
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()

    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()
    
    for i in range(len(topk_references)):
        
        nearest = topk_references[i][mask[i]][:neighbour_range]
    
        nearest_dict[query_labels[i].item()] = list(nearest)
    

    return nearest_dict
