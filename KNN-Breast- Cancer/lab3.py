import numpy as np
from lab3_utils import edit_distance, feature_names
from scipy import stats

# Hint: Consider how to utilize np.unique()

def process_inputs(inputs, m):
    new_inputs = []
    for x in inputs:
        row = []
       
        age = x[0]
        menopause = x[1]
        tumor_size = x[2]
        inv_nodes = x[3]
        node_caps = x[4]
        deg_malig = x[5]
        breast = x[6]
        quad = x[7]
        irradiat = x[8]
        #process age
        if age == '?':
            age = m[0][0][0]
        row.append(int(age.split('-')[0])/10)
        
        #process menopause
        
        if menopause == '?':
            menopause = m[0][1]
        if menopause == 'lt40':
            row.append(1)
            row.append(0)
            row.append(0)
        if menopause == 'ge40':
            row.append(0)
            row.append(1)
            row.append(0)
        if menopause == 'premeno':
            row.append(0)
            row.append(0)
            row.append(1)
        

        #process tumor size
        if tumor_size == '?':
            tumor_size = m[0][0][2]
        tumSol = int(tumor_size.split('-')[0])/5
        row.append(tumSol)

        #process inv nodes
        if inv_nodes == '?':
            inv_nodes= m[0][0][3]
        invSol = int(inv_nodes.split('-')[0])/3
        row.append(invSol)

        #process node caps
        nodeSol = 0
        if node_caps == '?':
            node_caps = m[0][0][4]
        if node_caps == 'yes':
            nodeSol = 1
        row.append(nodeSol)

        #process deg malig
        if deg_malig == '?':
            deg_malig = m[0][0][5]
        row.append(int(deg_malig)-1) # just to start at 0 

        #process breast
        breastSol = 0
        if breast == '?':
            breast = m[0][0][6]
        if breast == 'right':
            breastSol = 1
        row.append(breastSol)

        #process quad

        if quad == '?':
            quad = m[0][0][7]
        if quad == 'left_up':
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
        if quad == 'left_low':
            row.append(0)
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
        if quad == 'right_up':
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
            row.append(0)
        if quad == 'right_low':
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
        if quad == 'central':
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)

        #process irradiat

        irrSol = 0 

        if irradiat == '?':
            irradiat = m[0][0][8]
        if irradiat == 'yes':
            irrSol = 1
        row.append(irrSol)


        new_inputs.append(row)
    return new_inputs



def preprocess_data(training_inputs, testing_inputs, training_labels, testing_labels):
    
    processed_training_inputs, processed_testing_inputs = ([], [])
    processed_training_labels, processed_testing_labels = ([], [])
    # VVVVV YOUR CODE GOES ERE VVVVV $

    m = stats.mode(training_inputs)
    processed_training_inputs = process_inputs(training_inputs,m)
    processed_testing_inputs = process_inputs(testing_inputs, m)
    
    for x in training_labels:
        if( x == 'no-recurrence-events'):
            processed_training_labels.append(0)
        if (x == 'recurrence-events'):
            processed_training_labels.append(1)
    for y in testing_labels:
        if(y== 'no-recurrence-events'):
            processed_testing_labels.append(0)
        else:
            processed_testing_labels.append(1)

    # ^^^^^ YOUR CODE GOES ERE ^^^^^ $
    return processed_training_inputs, processed_testing_inputs, processed_training_labels, processed_testing_labels


# Hint: consider how to utilize np.argsort()
def k_nearest_neighbors(predict_on, reference_points, reference_labels, k, l, weighted):
    assert len(predict_on) > 0, f"parameter predict_on needs to be of length 0 or greater"
    assert len(reference_points) > 0, f"parameter reference_points needs to be of length 0 or greater"
    assert len(reference_labels) > 0, f"parameter reference_labels needs to be of length 0 or greater"
    assert len(reference_labels) == len(reference_points), f"reference_points and reference_labels need to be the" \
                                                           f" same length"
    predictions = []
    # VVVVV YOUR CODE GOES ERE VVVVV $
    patients = len(predict_on)
    
    #We will look at each patient we must make a prediction on
    for y in range(patients):
        patient_distances = []
        
        #Compare the inputs with all the inputs of the training set
        for x in range(len(reference_points)):
            
            #Caluculate the distance between the two
            dist = edit_distance(reference_points[x],predict_on[y],l)
            patient_distances.append((dist,reference_labels[x]))
        
        #Sort the distances calculated
        type1 = [('distance', float), ('reccurence', int)]
        pat_distance2 = np.array(patient_distances, dtype = type1)
        
        #Sort by distance and break ties by reccurrence
        Sorted_patient_distances = np.sort(pat_distance2, order = ['distance','reccurence'])
        
        #Go through the neighbors 
        recur = 0
        no_recur = 0 
        for z in range(k):

            #grab the labels of the sorted distances in range k
            label  = Sorted_patient_distances[z][1]
            
            #count up the total labels of reccurence and non reccurrence
            if (label == 0):
                no_recur +=1
            if (label == 1):
                recur +=1
        
        #decide which is the majority and tie break by recurrence
        if(recur>=no_recur):
            predictions.append(1)
        else:
            predictions.append(0)


    return predictions

